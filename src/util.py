from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


def get_info_from_html(html_path: str | Path) -> list[list[str]]:
    entry: list[list[str]] = []
    try:
        f = open(html_path, 'r')
    except (FileNotFoundError, IsADirectoryError) as e:
        logger.warning("Cannot read HTML file %s: %s", html_path, e)
        return entry
    with f:
        for l in f:
            l = l.strip()
            if l.startswith("<table>") or l.startswith("</table>"):
                if "<td>" not in l:
                    continue
                l = l.strip("<table>").strip("</table>")
            meta, img_src = l.rsplit("<td>", maxsplit=1)
            fname, expnum, *other = meta.split("<br>")
            img_path_fmt = '<td><img src="./images/{}.jpg"></tr>'
            img_src = img_path_fmt.format(fname.split(">")[-1])
            entry.append([fname, expnum, *other, img_src])
    return entry


def make_webpage(
    master_list: list[Any],
    pack_idx: list[int],
    root_dir: str | Path,
    base_name: str,
    num_element: int = 400,
) -> None:
    count = 0
    base_tmpl = "<table>\n{}\n</table>\n"
    content: list[str] = []
    start_exp = -1
    for i, idx in enumerate(pack_idx):
        if start_exp == -1:
            start_exp = master_list[idx][1]
        content.append("<br>".join(master_list[idx]))
        if num_element > 0 and i and i % num_element == 0:
            end_exp = master_list[idx][1]
            with open(root_dir / f"{count}_{base_name}_{start_exp}_{end_exp}.html", "w") as web:
                web.write(base_tmpl.format("\n".join(content)))
            count += 1
            start_exp = -1
            content = []
    if len(content):
        end_exp = master_list[idx][1]
        with open(root_dir / f"{count}_{base_name}_{start_exp}_{end_exp}.html", "w") as web:
            web.write(base_tmpl.format("\n".join(content)))
