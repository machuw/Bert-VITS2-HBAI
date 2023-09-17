import re

def keep_english_chinese(s):
    return re.sub(r"[^a-zA-Z\u4e00-\u9fa5]", "", s)

string = "#{REALNAME[ID(1)|HOSTONLY(true)]}"
result = keep_english_chinese(string)
print(result)  # 输出: "Hello你好"

