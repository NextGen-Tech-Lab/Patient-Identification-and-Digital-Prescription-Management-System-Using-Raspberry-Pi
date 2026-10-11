"""
test_hardware.py
----------------
Quick diagnostic script to test your LEDs and Buzzer connected to Pins 27-40.

Wiring:
- Pin 30 -> Ground (GND) for all components
- Pin 31 -> Green LED (GPIO 6) via 1.5k resistor
- Pin 33 -> Yellow LED (GPIO 13) via 1.5k resistor
- Pin 37 -> Red LED (GPIO 26) via 1.5k resistor
- Pin 32 -> Buzzer (+) (GPIO 12)
"""

import time
import sys

# Pin definitions (BCM numbering)
PIN_LED_GREEN  = 6    # Physical Pin 31
PIN_LED_YELLOW = 13   # Physical Pin 33
PIN_LED_RED    = 26   # Physical Pin 37
PIN_BUZZER     = 12   # Physical Pin 32

IS_RPI = False
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(PIN_LED_GREEN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_LED_YELLOW, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_LED_RED, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_BUZZER, GPIO.OUT, initial=GPIO.LOW)
    IS_RPI = True
    print("[HARDWARE] RPi.GPIO initialized successfully.")
except (ImportError, RuntimeError):
    print("[MOCK] Running in simulation mode (no Raspberry Pi hardware detected).")


def set_pin(pin, state):
    if IS_RPI:
        GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)


def test_leds_and_buzzer():
    print("\n--- Starting Hardware Test ---")
    print("All Grounds -> Pin 30")
    print("Testing Green LED (Pin 31 / GPIO 6)...")
    set_pin(PIN_LED_GREEN, True)
    time.sleep(1.0)
    set_pin(PIN_LED_GREEN, False)

    print("Testing Yellow LED (Pin 33 / GPIO 13)...")
    set_pin(PIN_LED_YELLOW, True)
    time.sleep(1.0)
    set_pin(PIN_LED_YELLOW, False)

    print("Testing Red LED (Pin 37 / GPIO 26)...")
    set_pin(PIN_LED_RED, True)
    time.sleep(1.0)
    set_pin(PIN_LED_RED, False)

    print("Testing Buzzer (Pin 32 / GPIO 12)...")
    for _ in range(3):
        set_pin(PIN_BUZZER, True)
        time.sleep(0.1)
        set_pin(PIN_BUZZER, False)
        time.sleep(0.1)

    print("\n[SUCCESS] Hardware test sequence completed!")


if __name__ == "__main__":
    try:
        test_leds_and_buzzer()
    finally:
        if IS_RPI:
            GPIO.cleanup()
