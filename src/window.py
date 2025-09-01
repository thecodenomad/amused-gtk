# window.py
#
# Copyright 2025 Ray
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gi.repository import Adw
from gi.repository import Gtk

from .frequency_display import FrequencyDisplay


@Gtk.Template(resource_path="/io/github/thecodenomad/amused_gtk/window.ui")
class AmusedGtkWindow(Adw.ApplicationWindow):
    __gtype_name__ = "AmusedGtkWindow"

    label = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Replace the simple label with the frequency display widget
        toolbar_view = self.label.get_parent()
        toolbar_view.set_content(None)  # Remove the current content

        # Create and add the frequency display
        self.frequency_display = FrequencyDisplay()
        toolbar_view.set_content(self.frequency_display)

        # Connect to destroy signal to properly disconnect from device
        self.connect("destroy", self.on_window_destroy)

    def on_window_destroy(self, widget):
        """Handle window destruction - properly disconnect from device"""
        print("🛑 Window closing - disconnecting from device...")

        # Stop streaming and disconnect from the frequency display
        if hasattr(self, 'frequency_display') and self.frequency_display:
            # Call the disconnect method
            self.frequency_display.disconnect_device()

        print("✅ Device disconnected successfully")
