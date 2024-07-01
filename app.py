# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import gradio as gr
import base64
from inference.qwen_infer import qwen_infer, PROMPT_TEMPLATE
from inference.clip_infer import clip_infer
from inference.dalle_infer import dalle_infer
from inference.sdxl_infer import sdxl_infer, STYLE_TEMPLATE, GENERAL_STYLE
from inference.I2VGen_infer import i2v_infer, v2v_infer
from inference.music_infer import music_infer
from inference.gpt_infer import gpt_infer
from inference.wanx_infer import wanx_infer
from functools import partial
from PIL import Image
import io

# Whether clear download model weights
parser = argparse.ArgumentParser()
parser.add_argument("--clear_cache", "--clear", action="store_true", default=False, help="Clean up downloaded model weights to save memory space")
clear_cache = parser.parse_args().clear_cache

def script_gen():
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=5>剧本生成(Script Generation)</center>""")
        with gr.Box():
            with gr.Row():
                with gr.Column(scale=1):
                    theme = gr.Textbox(label='主题(Theme)',
                                       placeholder='请输入剧本主题，如“未来科幻片“\n(Please enter the theme of the script, e.g., science fiction film.)',
                                       lines=2)
                    background = gr.Textbox(label='背景(Background)',
                                            placeholder='请输入剧本背景，如“城市”\n(Please enter the script background, e.g., space.)',
                                            lines=2)
                    scenario = gr.Textbox(label='剧情要求(Plot)',
                                          placeholder='请输入剧情要求，如“核战争爆发，充满想象力，跌宕起伏”\n(Please enter plot requirements, e.g., "imaginative, suspenseful ups and downs".)',
                                          lines=3)
                    language = gr.Radio(choices=['中文(Chinese)', '英文(English)'], label='语言(Language)',
                                        value='中文(Chinese)', interactive=True)
                    act = gr.Slider(minimum=1, maximum=6, value=3, step=1, interactive=True,
                                    label='剧本幕数(The number of scenes in the script)')
                    image = gr.Image(label='上传的图像(Uploaded Image)', source='upload', type='filepath')
                    with gr.Row():
                        clear_script = gr.Button('清空(Clear)')
                        submit_script = gr.Button('生成剧本(Submit)')
                with gr.Column(scale=2):
                    chat = gr.Chatbot(label='聊天对话(Chat)')
                    user_input = gr.Textbox(label='输入消息(Input)', placeholder='输入您的问题或请求...', lines=1)
                    send_button = gr.Button('发送(Send)')
                    chat_history = gr.State([])  # 用于存储聊天记录的状态变量
                    script_history = gr.State([])  # 用于存储脚本生成历史的状态变量

        def encode_image(image_path):
            # 读取图片
            image = Image.open(image_path)
            # 缩放图片
            image = image.resize((256, 256), Image.LANCZOS)
            # 转换为RGB格式（如果需要的话）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # 创建一个空字节流
            imgByteArr = io.BytesIO()
            # 将缩放后的图片保存到字节流
            image.save(imgByteArr, format='JPEG')
            # 获取字节流中的数据
            imgByteArr = imgByteArr.getvalue()

            return base64.b64encode(imgByteArr).decode("utf-8")

        def gpt_script(theme, background, act, scenario, language, image, history=None):
            fl = 0
            if image is None:
                inputs = PROMPT_TEMPLATE['script_onlytext'].format(theme=theme, background=background, act=act,
                                                                   scenario=scenario, language=language, image=None)
                fl = 0
            else:
                fl = 1
                if (theme is None) and (background is None) and (act is None) and (scenario is None) and (
                        language is None):
                    # inputs = PROMPT_TEMPLATE['script_onlyimg'].format(theme=theme, background=background, act=act,
                    #                                                   scenario=scenario, language=language,
                    #                                                   image=encode_image(image))
                    inputs = PROMPT_TEMPLATE['script_onlyimg'].format(theme=theme, background=background, act=act,
                                                                      scenario=scenario, language=language, image=image)
                else:
                    # inputs = PROMPT_TEMPLATE['script'].format(theme=theme, background=background, act=act,
                    #                                           scenario=scenario, language=language,
                    #                                           image=encode_image(image))
                    inputs = PROMPT_TEMPLATE['script'].format(theme=theme, background=background, act=act,
                                                              scenario=scenario, language=language)
            script, history = qwen_infer(inputs, image, has_img=fl, history=history)
            return script, history

        def edit_script(prompt, history=None):
            res, history = qwen_infer(prompt, has_img=0, history=history)
            return res, history

        def respond_to_user_message(user_message, theme, background, act, scenario, language, image, chat_history, script_history, fl=1):
            if fl == 0:
                response, new_script_history = gpt_script(theme, background, act, scenario, language, image)
            elif fl == 1:
                response, new_script_history = edit_script(user_message, script_history)
            # Update the chat history and script history
            chat_history.append([user_message, response])
            script_history = new_script_history

            # Return the new chat content, updated script history, and clear the user_input
            return chat_history, script_history, chat_history, ""

        submit_script.click(
            lambda theme, background, act, scenario, language, image, chat_history, script_history, fl=0: respond_to_user_message(
                "生成剧本", theme, background, act, scenario, language, image, chat_history, script_history, fl),
            inputs=[theme, background, act, scenario, language, image, chat_history, script_history],
            outputs=[chat, script_history, chat_history]
        )

        user_input.submit(
            lambda message, theme, background, act, scenario, language, image, chat_history, script_history, fl=1: respond_to_user_message(
                message, theme, background, act, scenario, language, image, chat_history, script_history, fl),
            inputs=[user_input, theme, background, act, scenario, language, image, chat_history, script_history],
            outputs=[chat, script_history, chat_history, user_input]
        )

        send_button.click(
            lambda message, theme, background, act, scenario, language, image, chat_history, script_history, fl=1: respond_to_user_message(
                message, theme, background, act, scenario, language, image, chat_history, script_history, fl),
            inputs=[user_input, theme, background, act, scenario, language, image, chat_history, script_history],
            outputs=[chat, script_history, chat_history, user_input]
        )

        clear_script.click(lambda: [None, None, 3, None, None, None, [], []],
                           inputs=[],
                           outputs=[theme, background, act, scenario, user_input, chat, chat_history, script_history],
                           queue=False)

    return demo

def production_still_gen():
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=5>剧照生成(Movie still Generation)</center>""")
        with gr.Box():
            gr.Markdown("""<left><font size=3>Step 1: 填入一幕剧本，然后点击“生成”，可以得到对应剧本的剧照场景描述和文生图提示词。</left>""")
            gr.Markdown("""<left><font size=3>Step 1: Enter a script scene, then click "Submit" to obtain the corresponding movie still scene description and the prompt.</left>""")
            with gr.Row():
                with gr.Column(scale=1):
                    script = gr.Textbox(label='剧本(Script)', placeholder='请输入剧本中的一幕\n(Please enter a scene from the script.)', lines=8)
                    language = gr.Radio(choices=['中文(Chinese)', '英文(English)'], label='语言(Language)', value='中文(Chinese)', interactive=True)
                    with gr.Row():
                        clear_prompt = gr.Button('清空(Clear)')
                        submit_prompt = gr.Button('生成(Submit)')
                with gr.Column(scale=2):
                    still_description = gr.Textbox(label='剧照描述(Movie still description) BY Qwen-7B-Chat', lines=5, interactive=False)
                    SD_prompt = gr.Textbox(label='提示词(Prompt) BY Qwen-7B-Chat', lines=5, interactive=False)

            def qwen_still(script, language):
                inputs = PROMPT_TEMPLATE['still'].format(script=script, language=language)
                qwen_infer_p = partial(qwen_infer, clear_cache=clear_cache)
                still_description = qwen_infer_p(inputs=inputs)
                return still_description

            def qwen_sd_prompt(still_description):
                inputs = PROMPT_TEMPLATE['SD'].format(still_description=still_description)
                qwen_infer_p = partial(qwen_infer, clear_cache=clear_cache)
                SD_prompt = qwen_infer_p(inputs=inputs)
                return SD_prompt

            submit_prompt.click(qwen_still,
                                inputs=[script, language],
                                outputs=[still_description]).then(qwen_sd_prompt, inputs=[still_description], outputs=[SD_prompt])
            clear_prompt.click(lambda: [None, None, None],
                        inputs=[],
                        outputs=[script, still_description, SD_prompt],
                        queue=False)

        with gr.Box():
            gr.Markdown("""<left><font size=3>Step 2: 复制上一步中得到的提示词，选择合适的风格和参数，点击“生成”，得到剧照。</left>""")
            gr.Markdown("""<left><font size=3>Step 2: Copy the prompt obtained in the Step 1, select appropriate styles and parameters, click "Submit" to obtain a movie still.</left>""")
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label='提示词(Prompts)',lines=3)
                    negative_prompt = gr.Textbox(label='负向提示词(Negative Prompts)',lines=3)
                    with gr.Row():
                        height = gr.Slider(512, 1024, 1024, step=128, label='高度(Height)', interactive=True)
                        width = gr.Slider(512, 1024, 1024, step=128, label='宽度(Width)', interactive=True)
                    with gr.Row():
                        scale = gr.Slider(1, 15, 10, step=.25, label='引导系数(CFG scale)', interactive=True)
                        steps = gr.Slider(25, maximum=100, value=50, step=5, label='迭代步数(Steps)', interactive=True)
                    seed = gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True, label='随机数种子(Seed)', interactive=True)

                with gr.Column(scale=3):
                    output_image = gr.Image(label='剧照(Movie still) BY SDXL-1.0', interactive=False, height=400)
                    with gr.Row():
                        clear = gr.Button('清空(Clear)')
                        submit = gr.Button('生成(Submit)')

                sdxl_infer_p = partial(sdxl_infer, clear_cache=clear_cache)
                submit.click(sdxl_infer_p, inputs=[prompt, negative_prompt, height, width, scale, steps, seed], outputs=output_image)
                clear.click(lambda: [None, None, 1024, 1024, 10, 50, None], inputs=[], outputs=[prompt, negative_prompt, height, width, scale, steps, output_image], queue=False)

            with gr.Accordion("提示词助手(Prompt assistant)", open=False):
                gr.Markdown("""<left><font size=2>您可以在此定制风格提示词，复制到提示词框中。固定风格提示词可以一定程度上固定剧照的风格。(You can customize your own style prompts here and copy them into the prompt textbox. Using consistent style prompts can help maintain a unified style of the movie stills to some degree.)</left>""")
                workbench = gr.Textbox(label="提示词工作台(Prompt workbench)", interactive=True, lines=2)
                general_style = gr.Radio(list(GENERAL_STYLE.keys()), value='无(None)', label='大致风格(General styles)', interactive=True)
                with gr.Row():
                    with gr.Column():
                        art = gr.Dropdown(STYLE_TEMPLATE['art'], multiselect=True, label="艺术风格(Art)")
                        atmosphere = gr.Dropdown(STYLE_TEMPLATE['atmosphere'], multiselect=True, label="场景氛围(Atmosphere)")
                        illustration_style = gr.Dropdown(STYLE_TEMPLATE['illustration style'], multiselect=True, label="插画风格(Illustration style)")
                    with gr.Column():
                        theme = gr.Dropdown(STYLE_TEMPLATE['theme'], multiselect=True, label="主题(Theme)")
                        image_quality = gr.Dropdown(STYLE_TEMPLATE['image quality'], multiselect=True, label="画质(Image quality)")
                        lighting = gr.Dropdown(STYLE_TEMPLATE['lighting'], multiselect=True, label="光照(Lighting)")
                    with gr.Column():
                        lens_style = gr.Dropdown(STYLE_TEMPLATE['lens style'], multiselect=True, label="相机镜头(Lens style)")
                        character_shot = gr.Dropdown(STYLE_TEMPLATE['character shot'], multiselect=True, label="人物镜头(Character shot)")
                        view = gr.Dropdown(STYLE_TEMPLATE['view'], multiselect=True, label="视角(View)")

                submit_style = gr.Button("提交至工作台(Submit to workbench)")

                STYLE_NAME = [general_style, art, atmosphere, illustration_style, theme, image_quality, lighting, lens_style, character_shot, view]
                def update_workbench(*styles):
                    style_prompt = GENERAL_STYLE[styles[0]]
                    style_list = []
                    for style in styles[1:]:
                        for word in style:
                            style_list.append(word)
                    style_prompt += ", ".join(style_list)
                    return style_prompt

                submit_style.click(fn=update_workbench,
                                    inputs=STYLE_NAME,
                                    outputs=[workbench],
                                    queue=False)

    return demo

def production_still_gen():
    with (gr.Blocks() as demo):
        gr.Markdown("""<center><font size=5>剧照生成(Movie still Generation)</center>""")
        with gr.Box():
            gr.Markdown(
                """<left><font size=3>Step 1: 填入一幕剧本，然后点击“生成”，可以得到对应剧本的剧照场景描述和文生图提示词。</left>""")
            with gr.Row():
                with gr.Column(scale=1):
                    script = gr.Textbox(label='剧本(Script)',
                                        placeholder='请输入剧本中的一幕\n(Please enter a scene from the script.)',
                                        lines=8)
                    language = gr.Radio(choices=['中文(Chinese)', '英文(English)'], label='语言(Language)',
                                        value='中文(Chinese)', interactive=True)
                    with gr.Row():
                        clear_prompt = gr.Button('清空(Clear)')
                        submit_prompt = gr.Button('生成(Submit)')

                with gr.Column(scale=2):
                    still_description = gr.Textbox(label='剧照描述(Movie still description) BY Qwen-Turbo', lines=5,
                                                   interactive=False)
                    SD_prompt = gr.Textbox(label='提示词(Prompt) BY Qwen-Turbo', lines=5, interactive=False)
                    # image_output = gr.Image(label='生成的图像(Generated Image)', type='auto')

            def gpt_still(script, language):
                inputs = PROMPT_TEMPLATE['still'].format(script=script, language=language)
                still_description, history = qwen_infer(inputs, '', 0)
                return still_description

            def gpt_sd_prompt(still_description):
                inputs = PROMPT_TEMPLATE['DALLE'].format(still_description=still_description)
                SD_prompt, history = qwen_infer(inputs, '', 0)
                return SD_prompt

            submit_prompt.click(gpt_still,
                                inputs=[script, language],
                                outputs=[still_description]
                               ).then(gpt_sd_prompt,
                                      inputs=[still_description],
                                      outputs=[SD_prompt]
                               )


            clear_prompt.click(lambda: [None, None, None],
                               inputs=[],
                               outputs=[script, still_description, SD_prompt],
                               queue=False)
        with gr.Box():
            gr.Markdown("""<left><font size=3>Step 2: 复制上一步中得到的提示词，选择合适的风格和参数，点击“生成”，得到剧照。</left>""")
            gr.Markdown("""<left><font size=3>Step 2: Copy the prompt obtained in the Step 1, select appropriate styles and parameters, click "Submit" to obtain a movie still.</left>""")
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label='提示词(Prompts)',lines=3)
                    negative_prompt = gr.Textbox(label='负向提示词(Negative Prompts)',lines=3)
                    with gr.Row():
                        height = gr.Slider(128, 1024, 1024, step=128, label='高度(Height)', interactive=True)
                        width = gr.Slider(128, 1024, 1024, step=128, label='宽度(Width)', interactive=True)
                    with gr.Row():
                        scale = gr.Slider(1, 15, 10, step=.25, label='引导系数(CFG scale)', interactive=True)
                        steps = gr.Slider(25, maximum=100, value=50, step=5, label='迭代步数(Steps)', interactive=True)
                    seed = gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True, label='随机数种子(Seed)', interactive=True)

                with gr.Column(scale=3):
                    output_image = gr.Image(label='剧照(Movie still) BY Wanx', interactive=False, height=400)
                    with gr.Row():
                        clear = gr.Button('清空(Clear)')
                        submit = gr.Button('生成(Submit)')

                wanx_infer_p = partial(wanx_infer)
                submit.click(wanx_infer_p, inputs=[prompt, negative_prompt, height, width, scale, steps, seed], outputs=output_image)
                clear.click(lambda: [None, None, 1024, 1024, 10, 50, None], inputs=[], outputs=[prompt, negative_prompt, height, width, scale, steps, output_image], queue=False)

            with gr.Accordion("提示词助手(Prompt assistant)", open=False):
                gr.Markdown("""<left><font size=2>您可以在此定制风格提示词，复制到提示词框中。固定风格提示词可以一定程度上固定剧照的风格。(You can customize your own style prompts here and copy them into the prompt textbox. Using consistent style prompts can help maintain a unified style of the movie stills to some degree.)</left>""")
                workbench = gr.Textbox(label="提示词工作台(Prompt workbench)", interactive=True, lines=2)
                general_style = gr.Radio(list(GENERAL_STYLE.keys()), value='无(None)', label='大致风格(General styles)', interactive=True)
                with gr.Row():
                    with gr.Column():
                        art = gr.Dropdown(STYLE_TEMPLATE['art'], multiselect=True, label="艺术风格(Art)")
                        atmosphere = gr.Dropdown(STYLE_TEMPLATE['atmosphere'], multiselect=True, label="场景氛围(Atmosphere)")
                        illustration_style = gr.Dropdown(STYLE_TEMPLATE['illustration style'], multiselect=True, label="插画风格(Illustration style)")
                    with gr.Column():
                        theme = gr.Dropdown(STYLE_TEMPLATE['theme'], multiselect=True, label="主题(Theme)")
                        image_quality = gr.Dropdown(STYLE_TEMPLATE['image quality'], multiselect=True, label="画质(Image quality)")
                        lighting = gr.Dropdown(STYLE_TEMPLATE['lighting'], multiselect=True, label="光照(Lighting)")
                    with gr.Column():
                        lens_style = gr.Dropdown(STYLE_TEMPLATE['lens style'], multiselect=True, label="相机镜头(Lens style)")
                        character_shot = gr.Dropdown(STYLE_TEMPLATE['character shot'], multiselect=True, label="人物镜头(Character shot)")
                        view = gr.Dropdown(STYLE_TEMPLATE['view'], multiselect=True, label="视角(View)")

                submit_style = gr.Button("提交至工作台(Submit to workbench)")

                STYLE_NAME = [general_style, art, atmosphere, illustration_style, theme, image_quality, lighting, lens_style, character_shot, view]
                def update_workbench(*styles):
                    style_prompt = GENERAL_STYLE[styles[0]]
                    style_list = []
                    for style in styles[1:]:
                        for word in style:
                            style_list.append(word)
                    style_prompt += ", ".join(style_list)
                    return style_prompt

                submit_style.click(fn=update_workbench,
                                    inputs=STYLE_NAME,
                                    outputs=[workbench],
                                    queue=False)

    return demo

def video_gen():
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=5>视频生成(Video Generation)</center>""")
        with gr.Box():
            gr.Markdown("""<left><font size=3>Step 1: 上传剧照（建议图片比例为1:1），然后点击“生成”，得到满意的视频后进行下一步。</left>""")
            gr.Markdown("""<left><font size=3>Step 1: Upload a movie still (it is recommended that the image ratio is 1:1), then click "Submit" to get a satisfactory video before moving to the Step 2.</left>""")
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(label="剧照(Movie still)", type="filepath", interactive=True, height=300)
                    with gr.Row():
                        clear_image = gr.Button("清空(Clear)")
                        submit_image = gr.Button("生成视频(Submit)")
                with gr.Column():
                    video_out_1 = gr.Video(label='视频(Video) BY I2VGen-XL', interactive=False, height=300)
        with gr.Box():
            gr.Markdown("""<left><font size=3>Step 2: 补充对视频内容的英文文本描述，然后点击“生成高分辨率视频”。</left>""")
            gr.Markdown("""<left><font size=3>Step 2: Add the English text description of the video you want to generate, then click "Submit".</left>""")
            with gr.Row():
                with gr.Column():
                    text_in = gr.Textbox(label="视频描述(Video description)", placeholder='请输入对视频场景的英文描述\n(Please enter a description of the video scene.)', lines=8)
                    with gr.Row():
                        clear_video = gr.Button("清空(Clear)")
                        submit_video = gr.Button("生成高分辨率视频(Submit)")
                with gr.Column():
                    video_out_2 = gr.Video(label='高分辨率视频(High-resolutions video) BY MS-Vid2Vid-XL', interactive=False, height=300)
        i2v_infer_p = partial(i2v_infer, clear_cache=clear_cache)
        v2v_infer_p = partial(v2v_infer, clear_cache=clear_cache)
        submit_image.click(i2v_infer_p, inputs=[image_in], outputs=[video_out_1])
        submit_video.click(v2v_infer_p, inputs=[video_out_1, text_in], outputs=[video_out_2])
        clear_image.click(lambda: [None, None], inputs=[], outputs=[image_in, video_out_1], queue=False)
        clear_video.click(lambda: [None, None], inputs=[], outputs=[text_in, video_out_2], queue=False)

    return demo


def music_gen():
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=5>音乐生成(Music Generation)</center>""")
        with gr.Row():
            with gr.Column():
                description = gr.Text(label="音乐描述(Music description)", interactive=True, lines=10)
                duration = gr.Slider(minimum=1, maximum=30, value=10, label="生成时长(Duration)", interactive=True)
                with gr.Row():
                    clear = gr.Button("清空(Clear)")
                    submit = gr.Button("生成(Submit)")
            with gr.Column():
                output = gr.Video(label="Music BY MusicGen", interactive=False)

            music_infer_p = partial(music_infer, clear_cache=clear_cache)
            submit.click(music_infer_p, inputs=[description, duration], outputs=[output])
            clear.click(lambda: ["small", None, 10, None], inputs=[], outputs=[description, duration, output], queue=False)

    return demo



with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>StoryAgent</center>""")
    gr.Markdown("# <center> <font size=4>\N{fire} [Github star it here](https://github.com/B1ueorange/StoryAgent/tree/main)</center>")
    with gr.Tabs():
        with gr.TabItem('剧本生成(Script Generation)'):
            script_gen()
        with gr.TabItem('剧照生成(Movie still Generation)'):
            production_still_gen()
        with gr.TabItem('视频生成(Video Generation)'):
            video_gen()
        with gr.TabItem('音乐生成(Music Generation)'):
            music_gen()

demo.queue(status_update_rate=1).launch(share=True)
