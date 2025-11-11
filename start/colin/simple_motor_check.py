#!/usr/bin/env python3
"""Simple motor identification script with verbose output."""

import sys
print("Script started!", flush=True)

try:
    import scservo_sdk as scs
    print("✓ scservo_sdk imported successfully", flush=True)
except ImportError as e:
    print(f"✗ Failed to import scservo_sdk: {e}", flush=True)
    sys.exit(1)

def identify_motor(port, motor_id):
    print(f"\n{'='*60}", flush=True)
    print(f"Attempting to identify motor at:", flush=True)
    print(f"  Port: {port}", flush=True)
    print(f"  Motor ID: {motor_id}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    protocol = 0
    baudrate = 1000000
    
    try:
        print(f"Creating port handler for {port}...", flush=True)
        port_handler = scs.PortHandler(port)
        
        print(f"Creating packet handler (protocol {protocol})...", flush=True)
        packet_handler = scs.PacketHandler(protocol)
        
        print(f"Opening port {port}...", flush=True)
        if not port_handler.openPort():
            print(f"✗ FAILED to open port {port}", flush=True)
            print("  Make sure:", flush=True)
            print(f"  - Port {port} is correct", flush=True)
            print("  - No other program is using the port", flush=True)
            print("  - You have permission to access the port", flush=True)
            return None
        
        print(f"✓ Port {port} opened successfully", flush=True)
        
        print(f"Setting baudrate to {baudrate}...", flush=True)
        if not port_handler.setBaudRate(baudrate):
            print(f"✗ Failed to set baudrate", flush=True)
            port_handler.closePort()
            return None
        
        print(f"✓ Baudrate set to {baudrate}", flush=True)
        
        # Try multiple baudrates
        baudrates_to_try = [1000000, 500000, 115200, 57600]
        
        for baud in baudrates_to_try:
            print(f"\nTrying baudrate {baud}...", flush=True)
            port_handler.setBaudRate(baud)
            
            print(f"Pinging motor ID {motor_id}...", flush=True)
            model_number, comm_result, error = packet_handler.ping(port_handler, motor_id)
            
            print(f"  Communication result: {comm_result}", flush=True)
            print(f"  Model number: {model_number}", flush=True)
            print(f"  Error: {error}", flush=True)
            
            if comm_result == scs.COMM_SUCCESS:
                print(f"\n{'='*60}", flush=True)
                print(f"✓ SUCCESS! Motor found!", flush=True)
                print(f"  Motor ID: {motor_id}", flush=True)
                print(f"  Model Number: {model_number}", flush=True)
                print(f"  Baudrate: {baud}", flush=True)
                
                # Identify motor type
                motor_types = {
                    777: "STS3215",
                    2825: "STS3250",
                    2057: "STS3235",
                    2569: "STS3095"
                }
                motor_type = motor_types.get(model_number, "UNKNOWN")
                print(f"  Motor Type: {motor_type}", flush=True)
                print(f"{'='*60}", flush=True)
                
                port_handler.closePort()
                return model_number
            else:
                result_str = packet_handler.getTxRxResult(comm_result)
                print(f"  Failed: {result_str}", flush=True)
        
        print(f"\n✗ Could not communicate with motor ID {motor_id} at any baudrate", flush=True)
        port_handler.closePort()
        return None
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple motor identification")
    parser.add_argument("--port", required=True, help="COM port (e.g., COM13)")
    parser.add_argument("--id", type=int, required=True, help="Motor ID to check")
    
    args = parser.parse_args()
    print(f"Arguments parsed: port={args.port}, id={args.id}", flush=True)
    
    identify_motor(args.port, args.id)
    
    print("\nScript finished.", flush=True)
