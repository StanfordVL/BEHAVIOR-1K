import logging
import os
import pprint
from typing import Callable, List, Optional, Union
from omnigibson.learning.utils.misc_utils import match_patterns


def to_scientific_str(value, precision: int = 1, capitalize: bool = False) -> str:
    """
    0.0015 -> "1.5e-3"
    """
    if value == 0:
        return "0"
    return f"{value:.{precision}e}".replace("e-0", "E-" if capitalize else "e-")


def pprint_(*objs, **kwargs):
    """
    Use pprint to format the objects
    """
    print(
        *[pprint.pformat(obj, indent=2) if not isinstance(obj, str) else obj for obj in objs],
        **kwargs,
    )


# ==================== Logging filters ====================
class ExcludeLoggingFilter(logging.Filter):
    """
    Usage: logging.getLogger('name').addFilter(
        ExcludeLoggingFilter(['info mess*age', 'Warning: *'])
    )
    Supports wildcard.
    https://relaxdiego.com/2014/07/logging-in-python.html
    """

    def __init__(self, patterns):
        super().__init__()
        self._patterns = patterns

    def filter(self, record):
        if match_patterns(record.msg, include=self._patterns):
            return False
        else:
            return True


def logging_exclude_pattern(
    logger_name,
    patterns: Union[str, list[str], Callable, list[Callable], None],
):
    """
    Args:
        patterns: see omnigibson.learning.utils.misc_utils.match_patterns
    """
    logging.getLogger(logger_name).addFilter(ExcludeLoggingFilter(patterns))


# coding: utf-8
# Copyright (c) 2008-2011 Volvox Development Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Original Author: Konstantin Lepa <konstantin.lepa@gmail.com>
# Updated by Jim Fan

STYLES = dict(
    list(
        zip(
            ["bold", "dark", "", "underline", "blink", "", "reverse", "concealed"],
            list(range(1, 9)),
        )
    )
)
del STYLES[""]


HIGHLIGHTS = dict(
    list(
        zip(
            ["grey", "red", "green", "yellow", "blue", "magenta", "cyan", "white"],
            list(range(40, 48)),
        )
    )
)


COLORS = dict(
    list(
        zip(
            ["grey", "red", "green", "yellow", "blue", "magenta", "cyan", "white"],
            list(range(30, 38)),
        )
    )
)


def _strip_bg_prefix(color):
    "on_red -> red"
    if color.startswith("on_"):
        return color[len("on_") :]
    else:
        return color


RESET = "\033[0m"


def color_text(
    text,
    color: Optional[str] = None,
    bg_color: Optional[str] = None,
    styles: Optional[Union[str, List[str]]] = None,
):
    """Colorize text.

    Available text colors:
        red, green, yellow, blue, magenta, cyan, white.

    Available text highlights:
        on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white.

    Available attributes:
        bold, dark, underline, blink, reverse, concealed.

    Example:
        colored('Hello, World!', 'red', 'on_grey', ['blue', 'blink'])
        colored('Hello, World!', 'green')
    """
    if os.getenv("ANSI_COLORS_DISABLED") is None:
        fmt_str = "\033[%dm%s"
        if color is not None:
            text = fmt_str % (COLORS[color], text)

        if bg_color is not None:
            bg_color = _strip_bg_prefix(bg_color)
            text = fmt_str % (HIGHLIGHTS[bg_color], text)

        if styles is not None:
            if isinstance(styles, str):
                styles = [styles]
            for style in styles:
                text = fmt_str % (STYLES[style], text)

        text += RESET
    return text
