"""
UDP communication test utility for pose imitation system.

Use this to test that commands are being sent correctly from the Python
pipeline to the Webots controller.

Usage:
    python test_udp_communication.py
"""
from __future__ import annotations

import json
import socket
import time
from typing import Dict


def test_receive_commands(port: int = 8765, timeout: float = 10.0) -> None:
    """
    Listen for and display pose commands from Python pipeline.
    
    Args:
        port: UDP port to listen on (default 8765)
        timeout: Timeout in seconds (default 10.0)
    """
    print(f"Starting UDP listener on port {port}...")
    print(f"Timeout: {timeout} seconds")
    print("Make sure the Python pipeline is running and sending to this address!")
    print("-" * 70)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(timeout)
    
    frame_count = 0
    total_bytes = 0
    start_time = time.time()
    
    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
                total_bytes += len(data)
                frame_count += 1
                
                payload = json.loads(data.decode("utf-8"))
                timestamp = payload.get("timestamp_s", 0)
                frame_idx = payload.get("frame_index", 0)
                angles = payload.get("joint_angles_rad", {})
                
                if frame_count == 1:
                    print(f"✓ Received first packet from {addr}")
                
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(
                        f"Frame {frame_idx:4d} | Timestamp {timestamp:.3f} | "
                        f"{len(angles):2d} joints | "
                        f"FPS: {fps:.1f} | Total: {frame_count} frames, {total_bytes} bytes"
                    )
                    
                    # Show joint details on first frame
                    if frame_count == 10:
                        print("\nSample joint angles (frame {}):".format(frame_idx))
                        for joint, angle in sorted(angles.items()):
                            print(f"  {joint:20s}: {angle:7.4f} rad ({angle*180/3.14159:7.2f}°)")
                        print()
                
            except socket.timeout:
                print("\n✗ No data received (timeout)")
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        sock.close()
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print("-" * 70)
        print(f"Summary:")
        print(f"  Frames received: {frame_count}")
        print(f"  Total bytes: {total_bytes}")
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Average packet size: {total_bytes / max(frame_count, 1):.0f} bytes")


def test_send_dummy_command(host: str = "127.0.0.1", port: int = 8765) -> None:
    """
    Send a dummy pose command for testing. Useful for verifying the
    Webots controller is listening.
    
    Args:
        host: Target host (default 127.0.0.1)
        port: Target port (default 8765)
    """
    print(f"Sending test command to {host}:{port}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    payload = {
        "timestamp_s": time.time(),
        "frame_index": 0,
        "joint_angles_rad": {
            "LShoulderPitch": 0.5,
            "RShoulderPitch": -0.5,
            "LElbowRoll": 0.8,
            "RElbowRoll": -0.8,
        }
    }
    
    message = json.dumps(payload).encode("utf-8")
    sock.sendto(message, (host, port))
    print(f"✓ Sent {len(message)} bytes")
    print(f"  Frame index: {payload['frame_index']}")
    print(f"  Joints: {len(payload['joint_angles_rad'])}")
    sock.close()


def test_bidirectional(port: int = 8765) -> None:
    """
    Test bidirectional communication (send a command, then listen for response).
    
    Args:
        port: UDP port (default 8765)
    """
    print(f"Testing bidirectional communication on port {port}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(2.0)
    
    # Send test command
    payload = {
        "timestamp_s": time.time(),
        "frame_index": 999,
        "joint_angles_rad": {"TorsoPitch": 0.3}
    }
    message = json.dumps(payload).encode("utf-8")
    sock.sendto(message, ("127.0.0.1", port))
    print(f"✓ Sent test command (frame 999)")
    
    # Try to receive it back (if loopback)
    try:
        data, addr = sock.recvfrom(65535)
        received = json.loads(data.decode("utf-8"))
        print(f"✓ Received response from {addr}")
        print(f"  Frame: {received.get('frame_index')}")
    except socket.timeout:
        print("✗ No response (expected if controller doesn't echo)")
    
    sock.close()


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Pose Imitation UDP Communication Tester")
    print("=" * 70)
    print()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "listen":
            test_receive_commands()
        elif mode == "send":
            test_send_dummy_command()
        elif mode == "bidirectional":
            test_bidirectional()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python test_udp_communication.py [listen|send|bidirectional]")
    else:
        print("Modes:")
        print("  listen         - Listen for commands from pipeline (default)")
        print("  send           - Send a test command")
        print("  bidirectional  - Test send+receive")
        print()
        print("Starting in listen mode (Ctrl+C to stop)...")
        print()
        test_receive_commands(timeout=30.0)
