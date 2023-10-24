import re

def remove_lines_with_numbers(filename):
    res = []
    with open(filename, encoding="utf-8") as fd:
        for line in fd.readlines():
            text = line.strip().split("|")[3]
            if not re.search(r'\d', text):
                res.append(line)
    return res

if __name__ == "__main__":
    # 示例
    text = """
    Hello,
    This is a sample text.
    Line with number 123.
    Another line.
    Yet another line with number 456.
    """

    filename = "/root/autodl-tmp/models/Bert-VITS2-HBAI/filelists/honkai.cleaned.enzh.bert0.list"
    result = remove_lines_with_numbers(filename)

    output_path = "/root/autodl-tmp/models/Bert-VITS2-HBAI/filelists/honkai.cleaned.enzh.bert0.list.new"

    with open(output_path, "w", encoding="utf-8") as fd:
        for line in result:
            fd.write(line)
