"""
hardware_io.py
--------------
Thread-safe hardware controller for LEDs and Buzzer on Raspberry Pi.
Optimized for 3.5" GPIO Touchscreen LCD (using free pins 27-40).

Pin Configuration (Physical & BCM):
- Physical Pin 30 -> Common GND (all cathodes and buzzer negative)
- Physical Pin 31 -> BCM GPIO 6  (Green LED: AUTHORIZED / DISPENSE READY)
- Physical Pin 33 -> BCM GPIO 13 (Yellow LED: CAUTION / SCANNING)
- Physical Pin 37 -> BCM GPIO 26 (Red LED: LOCKED / ALERT)
- Physical Pin 32 -> BCM GPIO 12 (Buzzer (+))
"""

import time
import threading

# BCM Pin Assignments
PIN_LED_GREEN  = 6    # Physical Pin 31
PIN_LED_YELLOW = 13   # Physical Pin 33
PIN_LED_RED    = 26   # Physical Pin 37
PIN_BUZZER     = 12   # Physical Pin 32

_IS_RPI = False
_gpio_lock = threading.Lock()

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (PIN_LED_GREEN, PIN_LED_YELLOW, PIN_LED_RED, PIN_BUZZER):
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    _IS_RPI = True
    print("[HARDWARE] RPi.GPIO initialized on Pins 30(GND), 31(GRN), 33(YEL), 37(RED), 32(BUZZ).")
except (ImportError, RuntimeError):
    # Running on Windows / non-RPi machine
    pass


class HardwareController:
    def __init__(self):
        self.is_rpi = _IS_RPI
        self.current_state = None

    def _write_pin(self, pin, state):
        if self.is_rpi:
            with _gpio_lock:
                GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)

    def all_leds_off(self):
        self._write_pin(PIN_LED_GREEN, False)
        self._write_pin(PIN_LED_YELLOW, False)
        self._write_pin(PIN_LED_RED, False)

    def beep(self, count=1, duration=0.1, pause=0.08):
        """Play buzzer beeps."""
        for _ in range(count):
            self._write_pin(PIN_BUZZER, True)
            time.sleep(duration)
            self._write_pin(PIN_BUZZER, False)
            time.sleep(pause)

    def chirp(self):
        """Very short tick for photo capture during enrollment."""
        def _run():
            self._write_pin(PIN_BUZZER, True)
            time.sleep(0.03)
            self._write_pin(PIN_BUZZER, False)
        threading.Thread(target=_run, daemon=True).start()

    def update_status(self, status: str, locked: bool = False):
        """
        Updates LED and Buzzer based on dispenser kiosk status:
        - LOCKED: Red LED ON + 3 alert beeps
        - AUTHORIZED: Green LED ON + 2 beeps
        - LOW_CONFIDENCE: Yellow LED ON + 1 warning beep
        - UNKNOWN / SCANNING: All LEDs OFF / quiet
        """
        state_key = "LOCKED" if locked else status
        if state_key == self.current_state:
            return  # Avoid repeated beeping for unchanged state

        self.current_state = state_key

        def _run_feedback():
            self.all_leds_off()

            if locked:
                self._write_pin(PIN_LED_RED, True)
                self.beep(count=3, duration=0.07, pause=0.06)
            elif status == "AUTHORIZED":
                self._write_pin(PIN_LED_GREEN, True)
                self.beep(count=2, duration=0.12, pause=0.08)
            elif status == "LOW_CONFIDENCE":
                self._write_pin(PIN_LED_YELLOW, True)
                self.beep(count=1, duration=0.25, pause=0.10)
            else:
                self.all_leds_off()

        threading.Thread(target=_run_feedback, daemon=True).start()

    def enroll_success(self):
        """Chime and flash green when patient face scan finishes."""
        def _run():
            self.all_leds_off()
            for _ in range(2):
                self._write_pin(PIN_LED_GREEN, True)
                self._write_pin(PIN_BUZZER, True)
                time.sleep(0.12)
                self._write_pin(PIN_LED_GREEN, False)
                self._write_pin(PIN_BUZZER, False)
                time.sleep(0.08)
        threading.Thread(target=_run, daemon=True).start()

    def cleanup(self):
        self.all_leds_off()
        self._write_pin(PIN_BUZZER, False)
        if self.is_rpi:
            with _gpio_lock:
                GPIO.cleanup()


# Global hardware singleton
hw = HardwareController()
