#!/usr/bin/env python3
"""
Dashboard Diagnostics Utility

This utility helps diagnose common issues with the Dask dashboard,
including WebSocket connection problems and port conflicts.
"""

import socket
import requests
import time
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def check_port_availability(port: int) -> bool:
    """Check if a port is available for use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False


def find_available_ports(start_port: int, count: int = 5) -> List[int]:
    """Find multiple available ports starting from start_port."""
    available_ports = []
    port = start_port
    
    while len(available_ports) < count and port < start_port + 100:
        if check_port_availability(port):
            available_ports.append(port)
        port += 1
    
    return available_ports


def check_dashboard_health(port: int, timeout: int = 5) -> Dict[str, Any]:
    """Check the health of a dashboard running on the specified port."""
    result = {
        'port': port,
        'is_accessible': False,
        'status_code': None,
        'response_time': None,
        'error': None
    }
    
    try:
        start_time = time.time()
        response = requests.get(f"http://localhost:{port}/status", timeout=timeout)
        end_time = time.time()
        
        result['is_accessible'] = True
        result['status_code'] = response.status_code
        result['response_time'] = round((end_time - start_time) * 1000, 2)  # ms
        
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection refused - dashboard not running"
    except requests.exceptions.Timeout:
        result['error'] = f"Request timeout after {timeout}s"
    except Exception as e:
        result['error'] = str(e)
    
    return result


def diagnose_dashboard_issues(port: int = 8888) -> Dict[str, Any]:
    """Comprehensive dashboard diagnostics."""
    logger.info(f"Running dashboard diagnostics for port {port}...")
    
    diagnostics = {
        'target_port': port,
        'port_available': check_port_availability(port),
        'dashboard_health': check_dashboard_health(port),
        'alternative_ports': find_available_ports(port, 5),
        'recommendations': []
    }
    
    # Generate recommendations
    if not diagnostics['port_available']:
        diagnostics['recommendations'].append(
            f"Port {port} is not available. Try one of these alternatives: {diagnostics['alternative_ports'][:3]}"
        )
    
    if not diagnostics['dashboard_health']['is_accessible']:
        diagnostics['recommendations'].append(
            f"Dashboard is not accessible: {diagnostics['dashboard_health']['error']}"
        )
        
        if diagnostics['dashboard_health']['error'] == "Connection refused - dashboard not running":
            diagnostics['recommendations'].append(
                "Start the Dask cluster to enable the dashboard"
            )
    
    if diagnostics['dashboard_health']['is_accessible']:
        diagnostics['recommendations'].append("Dashboard is healthy and accessible")
    
    return diagnostics


def print_diagnostics(diagnostics: Dict[str, Any]) -> None:
    """Print diagnostics in a human-readable format."""
    print("=" * 60)
    print("DASK DASHBOARD DIAGNOSTICS")
    print("=" * 60)
    print(f"Target Port: {diagnostics['target_port']}")
    print(f"Port Available: {'✓' if diagnostics['port_available'] else '✗'}")
    print()
    
    health = diagnostics['dashboard_health']
    print("Dashboard Health:")
    print(f"  Accessible: {'✓' if health['is_accessible'] else '✗'}")
    if health['is_accessible']:
        print(f"  Status Code: {health['status_code']}")
        print(f"  Response Time: {health['response_time']}ms")
    else:
        print(f"  Error: {health['error']}")
    print()
    
    print("Alternative Ports:")
    for port in diagnostics['alternative_ports'][:5]:
        print(f"  Port {port}: {'✓' if check_port_availability(port) else '✗'}")
    print()
    
    print("Recommendations:")
    for i, rec in enumerate(diagnostics['recommendations'], 1):
        print(f"  {i}. {rec}")
    print("=" * 60)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose Dask dashboard issues")
    parser.add_argument("--port", type=int, default=8888, help="Dashboard port to check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    diagnostics = diagnose_dashboard_issues(args.port)
    print_diagnostics(diagnostics)


if __name__ == "__main__":
    main()
