import time

def main():
    """Print 'Robot is testing' every 5 seconds indefinitely."""
    print("Starting robot test logging...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            print("Robot is testing")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nTest logging stopped.")

if __name__ == "__main__":
    main()
