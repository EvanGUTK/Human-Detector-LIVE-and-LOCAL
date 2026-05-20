"""Alert handling: sound, toast, flash state."""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field

from src.core.zones import ZoneEvent, ZoneEventType

logger = logging.getLogger(__name__)


@dataclass
class AlertState:
    flash_until: float = 0.0
    flash_zone_id: str | None = None
    total_alerts: int = 0
    recent_messages: list[str] = field(default_factory=list)

    def trigger_flash(self, zone_id: str, duration_ms: int) -> None:
        self.flash_until = time.time() + duration_ms / 1000.0
        self.flash_zone_id = zone_id

    def is_flashing(self) -> bool:
        return time.time() < self.flash_until


class AlertManager:
    def __init__(
        self,
        sound_enabled: bool = True,
        toast_enabled: bool = True,
        flash_ms: int = 800,
    ) -> None:
        self.sound_enabled = sound_enabled
        self.toast_enabled = toast_enabled
        self.flash_ms = flash_ms
        self.state = AlertState()

    def handle_events(self, events: list[ZoneEvent]) -> None:
        for ev in events:
            if ev.event_type != ZoneEventType.ENTER:
                continue
            self._fire_alert(ev.zone_id, ev.zone_name, ev.track_id, ev.identity)

    def test_alert(self, zone_id: str = "test", zone_name: str = "Test") -> None:
        """Trigger alert UI as if someone entered a zone (for testing)."""
        self._fire_alert(zone_id, zone_name, track_id=0, identity="Test")

    def _fire_alert(
        self, zone_id: str, zone_name: str, track_id: int, identity: str = ""
    ) -> None:
        self.state.total_alerts += 1
        if track_id == 0 and zone_name == "Test":
            msg = f"Test alert — zone '{zone_name}'"
        elif identity and identity not in ("Unknown", "Test"):
            msg = f"{identity} entered zone '{zone_name}'"
        else:
            msg = f"Unknown person entered zone '{zone_name}'"
        self.state.recent_messages.append(msg)
        if len(self.state.recent_messages) > 20:
            self.state.recent_messages.pop(0)
        self.state.trigger_flash(zone_id, self.flash_ms)
        if self.sound_enabled:
            threading.Thread(target=self._beep, daemon=True).start()
        if self.toast_enabled:
            threading.Thread(target=self._toast, args=(msg,), daemon=True).start()
        logger.info(msg)

    @staticmethod
    def _beep() -> None:
        if sys.platform == "win32":
            import winsound

            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        else:
            print("\a", end="")

    @staticmethod
    def _toast(message: str) -> None:
        try:
            from plyer import notification

            notification.notify(
                title="Person Detector",
                message=message,
                app_name="Person Detector",
                timeout=4,
            )
        except Exception as exc:
            logger.debug("Toast failed: %s", exc)
