from openai import OpenAI
# import openai
import base64

SCRIPT_TEMPLATE_ONLYTEXT = """你是一个编剧，请根据提供的短片主题、背景、幕数、剧情要求，设计一个剧本。剧本内容需要详细充分，每一幕不少于200字。
主题：{theme}
背景：{background}
幕数：{act}幕
剧情要求：{scenario}
语言：{language}
图片：{image}
"""

SCRIPT_TEMPLATE_ONLYIMG = """你是一个编剧，请根据提供的图片，设计一个剧本，请基于图片描述的内容，提供一个充满想象的虚构故事，故事内容需要与图像匹配。剧本内容需要详细充分，每一幕不少于200字。
主题：{theme}
背景：{background}
幕数：{act}幕
剧情要求：{scenario}
语言：{language}
图片：{image}
"""

SCRIPT_TEMPLATE = """你是一个编剧，请根据提供的短片主题、背景、幕数、剧情要求，以及图片中描述的内容，设计一个剧本。剧本内容需要详细充分，每一幕不少于200字。
主题：{theme}
背景：{background}
幕数：{act}幕
剧情要求：{scenario}
语言：{language}
图片：{image}
"""

STORY_TEMPLATE = """我会给你一个简单的图片描述，请提供一个充满想象的虚构故事，故事内容需要与图像匹配。
主题：{story_theme}
图片描述：{picture}"""

STILL_TEMPLATE = """根据剧本内容，设计一张剧照的场景描述，能体现剧本的核心内容。
剧本：{script}
语言：{language}"""

SD_PROMPT_TEMPLATE = """我们现在要通过stable diffusion进行图片生成，请根据场景描述，提炼出用于文本生成图像的英文prompt。
示例：
描述：一只美丽的蝴蝶在花丛中翩翩起舞，翅膀上闪烁着五彩斑斓的光芒，引来了勤劳的蜜蜂。蜜蜂在蝴蝶身边绕来绕去，试图吸引蝴蝶的注意。蝴蝶终于注意到了蜜蜂，停下来停歇在花朵上，与蜜蜂对视。
prompt：butterfly dancing in flower field, wings shimmering with rainbow colors, some bees flying around the butterfly, detailed realism, soft lighting, depth of field, 4k
描述：{still_description}
prompt："""

DALLE_PROMPT_TEMPLATE = """我们现在要进行图片生成，请根据场景描述，提炼出用于文本生成图像的英文prompt。
示例：
描述：一只美丽的蝴蝶在花丛中翩翩起舞，翅膀上闪烁着五彩斑斓的光芒，引来了勤劳的蜜蜂。蜜蜂在蝴蝶身边绕来绕去，试图吸引蝴蝶的注意。蝴蝶终于注意到了蜜蜂，停下来停歇在花朵上，与蜜蜂对视。
prompt：butterfly dancing in flower field, wings shimmering with rainbow colors, some bees flying around the butterfly, detailed realism, soft lighting, depth of field, 4k
描述：{still_description}
prompt："""

PROMPT_TEMPLATE = {
    "script": SCRIPT_TEMPLATE,
    "script_onlyimg": SCRIPT_TEMPLATE_ONLYIMG,
    "script_onlytext": SCRIPT_TEMPLATE_ONLYTEXT,
    "story":STORY_TEMPLATE,
    "still": STILL_TEMPLATE,
    "SD": SD_PROMPT_TEMPLATE,
    "DALLE": DALLE_PROMPT_TEMPLATE
}

def gpt_infer(inputs, has_img=0, history=None):
    messages=[]
    if history is None:
        messages.append({"role": "user", "content": "请你扮演一个编剧，完成撰写剧本等以下任务"})
    else:
        messages = history
    client = OpenAI(api_key="sk-proj-BunFOolIp5oBj4rOiOHhT3BlbkFJMQJxXT7E2RIdOUWWQfS3")

    # client = OpenAI(
    #     api_key="sk-XndM3qvksu9KBidoBd0f890fAc3f4192B37cD2BeF5D862Ca",  
    #     base_url= "https://apikeyplus.com/v1", 
    # )
    messages.append({"role": "user", "content": inputs})
    if has_img <= 0:
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    else:
        print(inputs)
        response = client.chat.completions.create(model="gpt-4o", messages=messages)

    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    return answer, messages
