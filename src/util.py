# util.py
# Utility functions for processing the images

def get_info_from_html(html_path):
    # parse the html webpage from Frank
    # html is in a table format
    # meta = []
    # img_src = []
    entry = []
    with open(html_path, 'r') as f:
        # first_entry = [] # meta
        # second_entry = [] # img_src
        count = 0
        for l in f:
            l = l.strip()
            # print(l)
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


def make_webpage(master_list, pack_idx, root_dir, base_name, num_element=400):
    count = 0
    base_tmpl = "<table>\n{}\n</table>\n"
    content = []
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
