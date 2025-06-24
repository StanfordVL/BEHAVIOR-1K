# Adapted from https://github.com/tocoteron/joycon-python/issues/16#issuecomment-1236055410

from pyjoycon import JoyCon, get_R_id, get_L_id
import sys
import time
import math

def clamp(x, min, max):
    if (x < min): return min;
    if (x > max): return max;
    return x;


# thanks to https://github.com/tocoteron/joycon-python/pull/27
class RumbleJoyCon(JoyCon):
    def __init__(self, *args, **kwargs):
        JoyCon.__init__(self, *args, **kwargs)

    def _send_rumble(self, data=b'\x00\x00\x00\x00\x00\x00\x00\x00'):
        self._RUMBLE_DATA = data
        self._write_output_report(b'\x10', b'', b'')

    def enable_vibration(self, enable=True):
        """Sends enable or disable command for vibration. Seems to do nothing."""
        self._write_output_report(b'\x01', b'\x48', b'\x01' if enable else b'\x00')

    def rumble_simple(self):
        """Rumble for approximately 1.5 seconds (why?). Repeat sending to keep rumbling."""
        self._send_rumble(b'\x98\x1e\xc6\x47\x98\x1e\xc6\x47')

    def rumble_stop(self):
        """Instantly stops the rumble"""
        self._send_rumble()


# derived from https://github.com/Looking-Glass/JoyconLib/blob/master/Packages/com.lookingglass.joyconlib/JoyconLib_scripts/Joycon.cs

class RumbleData:

    def set_vals(self, low_freq, high_freq, amplitude, time=0):
        self.h_f = high_freq
        self.amp = amplitude
        self.l_f = low_freq
        self.timed_rumble = False
        self.t = 0
        if time != 0:
            self.t = time / 1000.0
            self.timed_rumble = True

    def __init__(self, low_freq, high_freq, amplitude, time=0):
        self.h_f = None
        self.amp = None
        self.l_f = None
        self.t = None
        self.timed_rumble = None
        self.set_vals(low_freq, high_freq, amplitude, time)

    def GetData(self):
        rumble_data = [None] * 8
        if (self.amp == 0.0):
            rumble_data[0] = 0x0
            rumble_data[1] = 0x1
            rumble_data[2] = 0x40
            rumble_data[3] = 0x40
        else:
            l_f = clamp(self.l_f, 40.875885, 626.286133)
            amp = clamp(self.amp, 0.0, 1.0)
            h_f = clamp(self.h_f, 81.75177, 1252.572266)
            hf = int((round(32.0 * math.log(h_f * 0.1, 2)) - 0x60) * 4)
            lf = int(round(32.0 * math.log(l_f * 0.1, 2)) - 0x40)
            hf_amp = None
            if (amp == 0):
                hf_amp = 0
            elif amp < 0.01:
                hf_amp = 1
            elif amp < 0.117:
                hf_amp = int(((math.log(amp * 1000, 2) * 32) - 0x60) / (5 - pow(2, amp)) - 1)
            elif amp < 0.23:
                hf_amp = int(((math.log(amp * 1000, 2) * 32) - 0x60) - 0x5c)
            else:
                hf_amp = int((((math.log(amp * 1000, 2) * 32) - 0x60) * 2) - 0xf6)

            assert hf_amp is not None
            lf_amp = int(round(hf_amp) * .5)
            parity = int(lf_amp % 2)
            if (parity > 0):
                lf_amp -= 1

            lf_amp = int(lf_amp >> 1)
            lf_amp += 0x40
            if (parity > 0):
                lf_amp |= 0x8000

            rumble_data[0] = int(hf & 0xff)
            rumble_data[1] = int((hf >> 8) & 0xff)
            rumble_data[2] = lf
            rumble_data[1] += hf_amp
            rumble_data[2] += int((lf_amp >> 8) & 0xff)
            rumble_data[3] = int(lf_amp & 0xff)
        for i in range(4):
            rumble_data[4 + i] = rumble_data[i]
        # Debug.Log(string.Format("Encoded hex freq: {0:X2}", encoded_hex_freq));
        # Debug.Log(string.Format("lf_amp: {0:X4}", lf_amp));
        # Debug.Log(string.Format("hf_amp: {0:X2}", hf_amp));
        # Debug.Log(string.Format("l_f: {0:F}", l_f));
        # Debug.Log(string.Format("hf: {0:X4}", hf));
        # Debug.Log(string.Format("lf: {0:X2}", lf));
        return bytes(rumble_data)


if __name__ == "__main__":
    joycon_id_right = get_R_id()
    joycon_id_left = get_L_id()

    joyconR = RumbleJoyCon(*joycon_id_right)
    joyconL = RumbleJoyCon(*joycon_id_left)

    freq = 320
    amp = 0
    from IPython import embed; embed()
    while True:
        data = RumbleData(freq / 2, freq, amp)
        b = data.GetData()
        joyconR._send_rumble(b)
        time.sleep(1.5)
        joyconL._send_rumble(b)
        time.sleep(1.5)
        if amp > 0.9:
            amp = 0
        amp += 0.1
