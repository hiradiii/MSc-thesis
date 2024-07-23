import socket
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.log import setLogLevel
from scapy.all import send, IP, TCP, UDP, ICMP
import random
import time
from scapy.packet import Packet, bind_layers
from scapy.fields import ShortField, ByteField
from collections import defaultdict
import asyncio

# Define Modbus packets using Scapy
class ModbusMBAP(Packet):
    name = "ModbusMBAP"
    fields_desc = [
        ShortField("transaction_id", 1),
        ShortField("protocol_id", 0),
        ShortField("length", 6),
        ByteField("unit_id", 1)
    ]

class ModbusPDU(Packet):
    name = "ModbusPDU"
    fields_desc = [
        ByteField("function_code", 1)
    ]

bind_layers(TCP, ModbusMBAP, dport=502)
bind_layers(ModbusMBAP, ModbusPDU)

# Custom topology for the network
class SmartGridTopo(Topo):
    def build(self):
        # Add the utility server
        utility_server = self.addHost('us1', ip='10.0.0.1/24')
        # Add switches
        switches = [self.addSwitch(f'sw{i+1}', cls=OVSSwitch, failMode='secure', protocols='OpenFlow13') for i in range(10)]
        # Connect utility server to the first switch
        self.addLink(utility_server, switches[0])
        
        # Connect hosts to the remaining switches
        host_counter = 1
        for i in range(1, 10):
            num_hosts = 6 if i < 9 else 5  # The last switch gets 5 hosts
            for j in range(num_hosts):
                host_ip = f'10.{i}.{0}.{j+1}/24'
                host = self.addHost(f'h{host_counter}', ip=host_ip)
                self.addLink(host, switches[i])
                host_counter += 1


class TrafficFeature:
    def __init__(self, src_ip, dst_ip, src_port, dst_port, packet_count, packet_size, interval,
                 tcp_flags='', icmp_type=None, rst_count=0, seq_num=0, window_size=0, label=None,
                 tcp_syn_interval=None, icmp_interval=None, rst_interval=None, connection_start_time=None):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.packet_count = packet_count
        self.packet_size = packet_size
        self.interval = interval
        self.tcp_flags = tcp_flags
        self.tcp_syn_interval = tcp_syn_interval
        self.icmp_type = icmp_type
        self.icmp_interval = icmp_interval
        self.rst_count = rst_count
        self.rst_interval = rst_interval
        self.seq_num = seq_num
        self.window_size = window_size
        self.label = label
        self.connection_start_time = connection_start_time or time.time()

    def as_feature_vector(self):
        src_ip_int = int.from_bytes(socket.inet_aton(self.src_ip), 'big')
        dst_ip_int = int.from_bytes(socket.inet_aton(self.dst_ip), 'big')
        
        # Flags processing
        tcp_syn_flag = self.tcp_flags.count('S')
        tcp_rst_flag = self.tcp_flags.count('R')
        tcp_fin_flag = self.tcp_flags.count('F')
        total_tcp_packets = tcp_syn_flag + tcp_rst_flag + tcp_fin_flag
        # ICMP type handling
        icmp_echo_request = 1 if self.icmp_type == 8 else 0
        icmp_echo_reply = 1 if self.icmp_type == 0 else 0
        icmp_destination_unreachable = 1 if self.icmp_type == 3 else 0
        total_icmp_packets = icmp_echo_request + icmp_echo_reply + icmp_destination_unreachable
        icmp_ratio = icmp_echo_request / total_icmp_packets if total_icmp_packets > 0 else 0

        # total_flags = len(self.tcp_flags)
        syn_ratio = tcp_syn_flag / total_tcp_packets if total_tcp_packets > 0 else 0
        rst_ratio = tcp_rst_flag / total_tcp_packets if total_tcp_packets > 0 else 0

        # Average packet size calculation
        average_packet_size = self.packet_size / self.packet_count if self.packet_count > 0 else 0

        # Replace None values with 0 for intervals
        tcp_syn_interval = self.tcp_syn_interval if self.tcp_syn_interval is not None else 0
        icmp_interval = self.icmp_interval if self.icmp_interval is not None else 0
        rst_interval = self.rst_interval if self.rst_interval is not None else 0

        # Feature vector composition
        return [
            src_ip_int, dst_ip_int, self.src_port, self.dst_port, self.packet_count,
            syn_ratio, icmp_ratio, rst_ratio, tcp_syn_flag, tcp_rst_flag, tcp_fin_flag, icmp_echo_request, icmp_echo_reply,
            icmp_destination_unreachable, average_packet_size, self.interval, tcp_syn_interval,
            icmp_interval, rst_interval, self.seq_num, self.window_size,
            self.connection_start_time, self.label
        ]

    
class Metrics:
    def __init__(self):
        self.TP = 0  # True Positives
        self.TN = 0  # True Negatives
        self.FP = 0  # False Positives
        self.FN = 0  # False Negatives
        self.precision_rates = []
        self.specificity_rates = []
        self.detection_rates = []  # List to store detection rates for averaging
        self.fpr_rates = []
        self.fnr_rates = []

    def update_metrics(self, predicted_label, actual_label):
        if predicted_label == 1 and actual_label == 1:
            self.TP += 1
        elif predicted_label == 0 and actual_label == 0:
            self.TN += 1
        elif predicted_label == 1 and actual_label == 0:
            self.FP += 1
        elif predicted_label == 0 and actual_label == 1:
            self.FN += 1

        if (self.TP + self.FP) > 0:
            self.precision_rates.append(self.TP / (self.TP + self.FP))
        if (self.TN + self.FP) > 0:
            self.specificity_rates.append(self.TN / (self.TN + self.FP))
        if (self.TP + self.FN) > 0:
            self.detection_rates.append(self.TP / (self.TP + self.FN))
        if (self.FP + self.TN) > 0:
            self.fpr_rates.append(self.FP / (self.FP + self.TN))
        if (self.FN + self.TP) > 0:
            self.fnr_rates.append(self.FN / (self.FN + self.TP))

    def report(self):
        # Calculate final metrics
        precision = sum(self.precision_rates) / len(self.precision_rates) if self.precision_rates else 0
        specificity = sum(self.specificity_rates) / len(self.specificity_rates) if self.specificity_rates else 0
        detection_rate = sum(self.detection_rates) / len(self.detection_rates) if self.detection_rates else 0
        fpr = sum(self.fpr_rates) / len(self.fpr_rates) if self.fpr_rates else 0
        fnr = sum(self.fnr_rates) / len(self.fnr_rates) if self.fnr_rates else 0
        return (f"Precision: {precision * 100:.2f}%, Specificity: {specificity * 100:.2f}%, "
                f"Detection Rate: {detection_rate * 100:.2f}%, FPR: {fpr * 100:.2f}%, FNR: {fnr * 100:.2f}%")

    def calculate_average_metrics(self):
        # Calculate and return the average of all metrics
        precision_avg = sum(self.precision_rates) / len(self.precision_rates) if self.precision_rates else 0
        specificity_avg = sum(self.specificity_rates) / len(self.specificity_rates) if self.specificity_rates else 0
        detection_rate_avg = sum(self.detection_rates) / len(self.detection_rates) if self.detection_rates else 0
        fpr_avg = sum(self.fpr_rates) / len(self.fpr_rates) if self.fpr_rates else 0
        fnr_avg = sum(self.fnr_rates) / len(self.fnr_rates) if self.fnr_rates else 0
        return {
            "precision_avg": precision_avg,
            "specificity_avg": specificity_avg,
            "detection_rate_avg": detection_rate_avg,
            "fpr_avg": fpr_avg,
            "fnr_avg": fnr_avg
        }
    
class DDoSDetector:
    def __init__(self, k=3, n_init=10):
        self.k = k
        self.kmeans = KMeans(n_clusters=k, n_init=n_init)
        self.traffic_data = []
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.scaler = StandardScaler()
        self.metrics = Metrics()

    def add_traffic_data(self, feature_vector):
        self.traffic_data.append(feature_vector)

    def preprocess_data(self):
        numeric_data = [data[:-1] for data in self.traffic_data]
        numeric_data = np.array(numeric_data, dtype=float)
        if np.isnan(numeric_data).any():
            numeric_data = self.imputer.fit_transform(numeric_data)
        return self.scaler.fit_transform(numeric_data)

    def analyze_traffic(self):
        data = self.preprocess_data()
        return self.kmeans.fit_predict(data)

    def detect_ddos(self, labels):
        traffic_clusters = defaultdict(list)
        attack_detected = False
        current_time = time.time()
        for idx, label in enumerate(labels):
            traffic_clusters[label].append(self.traffic_data[idx])

        for cluster, features in traffic_clusters.items():
            src_ips = set(f[0] for f in features)
            
            # Corrected TCP and ICMP packet counts
            total_tcp_packets = sum(1 for f in features if f[8] or f[9] or f[10])  # Count any TCP flags
            total_icmp_packets = sum(1 for f in features if f[11] or f[12] or f[13])  # Count any ICMP requests

            tcp_syn_flags = sum(1 for f in features if f[8])  # Count SYN flags
            tcp_rst_flags = sum(1 for f in features if f[9])  # Count RST flags
            
            syn_ratio = tcp_syn_flags / total_tcp_packets if total_tcp_packets > 0 else 0
            rst_ratio = tcp_rst_flags / total_tcp_packets if total_tcp_packets > 0 else 0
            icmp_ratio = sum(f[6] for f in features) / total_icmp_packets if total_icmp_packets > 0 else 0
            
            tcp_syn_intervals = [f[16] for f in features if f[16] > 0]
            tcp_syn_interval_avg = sum(tcp_syn_intervals) / len(tcp_syn_intervals) if tcp_syn_intervals else 0
            
            rst_intervals = [f[18] for f in features if f[18] > 0]
            rst_interval_avg = sum(rst_intervals) / len(rst_intervals) if rst_intervals else 0
            
            icmp_intervals = [f[17] for f in features if f[17] > 0]
            icmp_interval_avg = sum(icmp_intervals) / len(icmp_intervals) if icmp_intervals else 0

            active_connections = sum(1 for f in features if f[-1] and (current_time - f[-2]) < 15)
            zero_window_packets = sum(1 for f in features if f[20] == 0)  # Count zero window size packets


            socket_conditions = [rst_ratio > 0.65, len(src_ips) >= 2, active_connections >= 20, zero_window_packets > 55]

            if all(socket_conditions):
                attack_type = "Socket Stress Attack"
                predicted_label = 1
            else:
                attack_type = "Normal"
                predicted_label = 0

            print(f"Cluster {cluster}: {attack_type} Detected with {len(features)} packets" if predicted_label == 1 else f"Cluster {cluster}: Normal traffic with {len(features)} packets")
            for feature in features:
                actual_label = feature[-1]
                self.metrics.update_metrics(predicted_label, actual_label)
                if predicted_label:
                    attack_detected = True

        if attack_detected:
            print(self.metrics.report())
        else:
            print("No DDoS attacks detected across all clusters.")

async def generate_traffic(net, src_host, dst_ip, dst_port, count, attack_type=None):
    src_device = net.get(src_host)  # Get the source device from the network based on the hostname

    for _ in range(count):
        current_time = time.time()
        
        if attack_type == "Socket Stress Attack":
            tcp_flags = "R"
            seq_num = random.randint(0, 0xFFFFFFFF)
            window_size = 0
            interval = random.uniform(0.01, 0.05)  # Very short intervals for high stress
            pkt = IP(src=src_device.IP(), dst=dst_ip) / TCP(sport=random.randint(1024, 65535), dport=dst_port, flags=tcp_flags, seq=seq_num, window=window_size)
            send(pkt, verbose=False)  # Send the packet silently
            await asyncio.sleep(interval)  # Brief pause to mimic realistic traffic
        else:
            # Handling normal UDP traffic
            interval = 0.4  # Standard interval for normal UDP traffic
            pkt = IP(src=src_device.IP(), dst=dst_ip) / UDP(sport=random.randint(1024, 65535), dport=dst_port)
            send(pkt, verbose=False)  # Send the packet silently
            await asyncio.sleep(interval)  # Fixed interval for normal traffic
    
def gather_traffic_data(net):
    traffic_features = []  # Store traffic data as feature vectors
    last_rst_packet_time = {}  # Store the last RST packet time for each host
    connection_times = {}  # Store start times for active connections, especially for attacks

    for host in net.hosts:
        src_ip = host.IP()
        dst_ip = "10.0.0.1"
        src_port = random.randint(1024, 65535)
        dst_port = 502
        packet_count = random.randint(1, 10)
        packet_sizes = [random.randint(40, 1500) for _ in range(packet_count)]
        current_time = time.time()

        attack_type = random.choices(
            ['Normal', 'Socket Stress Attack'],
            weights=[0.85, 0.15],  # Adjusted weights for normal and stress attack scenarios
            k=1)[0]

        tcp_flags = ""
        label = 0
        rst_interval = None
        window_size = 9000  # Default large window size for normal traffic
        connection_start_time = None

        if attack_type == "Socket Stress Attack":
            tcp_flags = "R"  # Using RST flag to indicate the nature of the packet in the context of this attack
            window_size = 0  # Set window size to zero to simulate the Sockstress attack
            if src_ip in last_rst_packet_time:
                rst_interval = current_time - last_rst_packet_time[src_ip]
            last_rst_packet_time[src_ip] = current_time
            connection_start_time = connection_times.get(src_ip, current_time)
            connection_times[src_ip] = current_time
            label = 1  # Mark as attack traffic
        else:
            # Handling normal UDP traffic or undefined types
            connection_start_time = None  # No connection time tracking for normal traffic
            label = 0  # Normal traffic

        packet_size = sum(packet_sizes)
        interval = 0.4  # Standard interval for all traffic types

        feature = TrafficFeature(src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port,
                                 packet_count=packet_count, packet_size=packet_size, interval=interval,
                                 tcp_flags=tcp_flags, window_size=window_size, rst_interval=rst_interval,
                                 label=label, connection_start_time=connection_start_time)
        traffic_features.append(feature.as_feature_vector())

    return traffic_features


async def monitor_network(detector, net, monitoring_event):
    while not monitoring_event.is_set():
        try:
            traffic_features = gather_traffic_data(net)  # Function to simulate and gather network traffic data
            for feature in traffic_features:
                detector.add_traffic_data(feature)  # Add each feature vector to the detector

            labels = detector.analyze_traffic()  # Analyze the aggregated traffic data and classify using k-means clustering
            if labels is not None and len(labels) > 0:
                detector.detect_ddos(labels)  # Detect potential DDoS attacks based on the analysis results
        except Exception as e:
            print(f"Error during network monitoring: {e}")  # Log and handle errors gracefully

        await asyncio.sleep(5)  # Asynchronous sleep for 5 seconds before next cycle

async def main():
    setLogLevel('info')
    topo = SmartGridTopo()
    net = Mininet(topo=topo, switch=OVSSwitch, controller=RemoteController('c0', ip='127.0.0.1', port=6633), autoSetMacs=True)
    net.start()

    # Initialize the DDoS detector with the number of clusters and initialization trials
    detector = DDoSDetector(k=3)

    # Create tasks for generating different types of traffic and monitoring the network
    task1 = asyncio.create_task(generate_traffic(net, 'h1', '10.0.0.1', random.randint(1024, 65535), 6250))  # Normal UDP traffic
    task2 = asyncio.create_task(generate_traffic(net, 'h2', '10.0.0.1', 502, 5000, "Socket Stress Attack"))  # TCP attack
    task3 = asyncio.create_task(generate_traffic(net, 'h3', '10.0.0.1', 502, 5000, "Socket Stress Attack"))  # TCP attack
    monitoring_event = asyncio.Event()

    # Monitoring network in parallel
    monitor_task = asyncio.create_task(monitor_network(detector, net, monitoring_event))

    # Waiting for all traffic generation tasks to complete before stopping the network
    await asyncio.gather(task1, task2, task3)

    # Signal the monitoring task to stop
    monitoring_event.set()

    # Ensure monitoring task stops
    await monitor_task

    # Output the final average detection rate after all tasks are completed
    avg_metrics = detector.metrics.calculate_average_metrics()
    print(f"Average Metrics: Precision: {avg_metrics['precision_avg'] * 100:.2f}%, "
          f"Specificity: {avg_metrics['specificity_avg'] * 100:.2f}%, "
          f"Detection Rate: {avg_metrics['detection_rate_avg'] * 100:.2f}%, "
          f"FPR: {avg_metrics['fpr_avg'] * 100:.2f}%, FNR: {avg_metrics['fnr_avg'] * 100:.2f}%")

    net.stop()

if __name__ == '__main__':
	asyncio.run(main())