import numpy as np
import whisper
import re
import srt
import cv2
import os
import random
from datetime import timedelta
from pydub import AudioSegment
from moviepy.editor import (
    VideoClip, AudioFileClip, CompositeAudioClip, VideoFileClip,
    TextClip, CompositeVideoClip, vfx, ImageClip
)
import scipy.io.wavfile as wavfile
import spacy
import pandas as pd
import librosa

#感情辞書の準備

print("[INFO] 💡 感情辞書を構築中...")
#GiNZA日本語モデル
nlp = spacy.load("ja_ginza")

#SNOWD18読み込み
snow_d18_path = 'D18-2018.7.24.xlsx'
snow_d18_df = pd.read_excel(snow_d18_path, sheet_name="作業者A")

snow_d18_mapping = {
    "好": "喜", "親": "喜",
    "安": "楽", "尊": "楽",
    "怒": "怒", "嫌": "怒",
    "悲": "哀", "不": "哀"
}

snow_d18_dict = {}
for index, row in snow_d18_df.iterrows():
    word = row['Word']
    emotions = list(str(row['Emotion']))
    for emo in emotions:
        mapped = snow_d18_mapping.get(emo)
        if mapped:
            snow_d18_dict[word] = mapped
            break

'''
# 日本語評価極性辞書（PN Dictionary）
pn_dict_path = 'pn.csv.m3.120408.trim'
pn_df = pd.read_csv(pn_dict_path, sep='\t', header=None, names=['word', 'polarity', 'annotation'])

pn_dict = {}
for index, row in pn_df.iterrows():
    if row['polarity'] == 'p':
        pn_dict[row['word']] = '喜'
    elif row['polarity'] == 'n':
        pn_dict[row['word']] = '哀'
'''

# 統合感情辞書
emotion_dict = {**snow_d18_dict}
#emotion_dict = {**snow_d18_dict, **pn_dict}

def get_emotion_from_dict(word):
    return emotion_dict.get(word, "なし")

def analyze_sentiment_per_word(text):
    doc = nlp(text)
    word_emotions = []
    for token in doc:
        base_word = token.lemma_
        emotion = get_emotion_from_dict(base_word)
        word_emotions.append((token.text, emotion))
    return word_emotions

def get_dominant_emotion(text):
    emotions = [emo for _, emo in analyze_sentiment_per_word(text) if emo != "なし"]
    if emotions:
        return max(set(emotions), key=emotions.count)
    return "なし"


#Whisperでの使った音声認識&字幕作成

def generate_subtitles(audio_path, output_path, offset_seconds=0, model_size="large"):
    print("📜 Whisperモデルをロード中...")
    model = whisper.load_model(model_size)
    print(f"🎤 音声ファイル '{audio_path}' を文字起こし中...")
    result = model.transcribe(audio_path)

    print(f"📂 字幕を '{output_path}' に保存中...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start = timedelta(seconds=int(segment["start"]) + offset_seconds)
            end = timedelta(seconds=int(segment["end"]) + offset_seconds)
            start_str = str(start)
            end_str = str(end)
            text = segment["text"].strip()
            start_srt = re.sub(r"\.(\d{3})\d{3}", r",\1", start_str)
            end_srt = re.sub(r"\.(\d{3})\d{3}", r",\1", end_str)
            f.write(f"{i}\n{start_srt} --> {end_srt}\n{text}\n\n")
    print("✅ 字幕生成が完了しました！")

#字幕アニメーション（感情付き）
    
def apply_advanced_animation(txt_clip, video_width, video_height, bpm, emotion="なし"):
    txt_clip = txt_clip.fx(vfx.fadein, 0.2).fx(vfx.fadeout, 0.2)
    txt_clip = txt_clip.set_opacity(1)

    # BPMによる基礎リズム周波数
    bpm_freq = (bpm / 60) * 2 * np.pi  # ≒ 12

    # ----------- ベース (常時揺れ) -----------
    # 全ての字幕がBPMに合わせて上下に小さくゆれる
    txt_clip = txt_clip.set_position(lambda t: ('center', video_height * 0.8 + 5 * np.sin(bpm_freq * t)))

    # ----------- 感情別追加アニメーション -----------
    emotion_effects = {
        "喜": [  # 拍に合わせてゆっくりスケール
            lambda clip: clip.fx(vfx.resize, lambda t: 1 + 0.03 * np.sin(bpm_freq * t))
        ],
        "怒": [  # 拍に合わせた左右小刻み振動
            lambda clip: clip.set_position(lambda t: (
                video_width // 2 - clip.w // 2 + 5 * np.sin(bpm_freq * 2 * t),
                video_height * 0.8 - clip.h // 2 + 5 * np.sin(bpm_freq * t)
            ))
        ],
        "哀": [  # 下げつつ鼓動のように少しスケール
            lambda clip: clip.set_position(lambda t: ('center', video_height * 0.82 + 3 * np.sin(bpm_freq * 0.5 * t))),
            lambda clip: clip.fx(vfx.resize, lambda t: 1 + 0.01 * np.sin(bpm_freq * 0.5 * t))
        ],
        "楽": [  # BPMに合わせた軽い左右スイング＋小回転
            lambda clip: clip.set_position(lambda t: (video_width//2 + 10 * np.sin(bpm_freq * t), video_height * 0.8 + 5 * np.sin(bpm_freq * t))),
            lambda clip: clip.rotate(lambda t: 2 * np.sin(bpm_freq * t))
        ]
    }

    for effect in emotion_effects.get(emotion, []):
        txt_clip = effect(txt_clip)

    return txt_clip






def create_subtitle_clips(subtitles, offvocal_path, video_width=1920, video_height=1080):
    print("[INFO] 🎬 感情に応じた字幕アニメーションを生成中...")
    subtitle_clips = []
    font_size = int(video_height * 0.06)
    
    #bpm解析
    bpm = get_bpm(offvocal_path)
    
    #字幕情報とりだし
    for sub in subtitles:
        start_time = sub.start.total_seconds()
        end_time = sub.end.total_seconds()
        duration = end_time - start_time
        text = sub.content.replace('\n', ' ')

        emotion = get_dominant_emotion(text)

        # ---- ① 影 ----
        shadow_clip = TextClip(
            text,
            fontsize=font_size,
            font="/Library/Fonts/cinecaption226.ttf",
            color="gray",
            stroke_color=None,
            size=(video_width * 0.8, None),
            method="label"
        ).set_duration(duration).set_position((2, 2))  # ここだけ相対的にずらす

        # ---- ② 本体 ----
        txt_clip = TextClip(
            text,
            fontsize=font_size,
            font="/Library/Fonts/cinecaption226.ttf",
            color="black",
            stroke_color="black",
            stroke_width=3,
            size=(video_width * 0.8, None),
            method="label"
        ).set_duration(duration)

        # ---- ③ 影＋本体を合成 (位置は決めない)
        combo_clip = CompositeVideoClip([shadow_clip, txt_clip]).set_duration(duration)
        
        # ↓↓ combo_clipにだけapply_advanced_animationをする
        combo_clip = apply_advanced_animation(combo_clip, video_width, video_height, bpm, emotion)
        combo_clip = combo_clip.set_start(start_time)

        subtitle_clips.append(combo_clip)

    return subtitle_clips






#最終動画合成関数

def create_final_video(video, subtitles, composite_audio, output_path):
    print(f"[INFO] 🎥 最終動画を合成中: {output_path}")
    video = video.resize((1920, 1080))
    video_duration = video.duration
    composite_audio = composite_audio.set_duration(video_duration)

    # 画像読み込み
    logo = (
    ImageClip("logo.png")
    .set_duration(175)
    .resize(height=700)
    .set_position(("center", "center"))
    .set_start(0)
    # .fadein(0.5).fadeout(0.5)
)

    #合成（動画＋字幕＋画像）
    elements = [video] + subtitles + [logo]

    final_clip = (CompositeVideoClip(elements, use_bgclip=True)
                  .set_duration(video_duration)
                  .set_audio(composite_audio))

    final_clip.write_videofile(output_path, fps=video.fps, codec="libx264", audio_codec="aac")
    

#変数定義
def get_bpm(filename):
    sampling_rate = 44100                                           #サンプリングレート
    
    #演算
    y, sr = librosa.load(path=filename, sr = sampling_rate)         #音声ファイルを1次元のNumPy浮動小数点配列（変数y）に格納
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)        #テンポを算出
    
    #結果表示
    #print('Estimated tempo: {:.2f} beats per minute'.format(tempo)) #結果を表示
    print(tempo)

    return 115


# -----------------------------------------
# 🚀 メイン処理
# -----------------------------------------

def main():
    vocal_path = "vocal.mp3"
    offvocal_path = "yoidore_offvocal.wav"
    temp_wav_path = "composite_audio.wav"
    video_path = "output_video_odiospe2.mp4"
    srt_path = "output_subtitles.srt"
    output_video = "final_output.mp4"
    offset_seconds = 8.695
    model_size = "medium"
    
    
    print("🎵 音声ファイルを合成中...")
    vocal_audio = AudioFileClip(vocal_path).volumex(1.2)
    offvocal_audio = AudioFileClip(offvocal_path).volumex(1.0)
    duration = min(vocal_audio.duration, offvocal_audio.duration)
    composite_audio = CompositeAudioClip([vocal_audio, offvocal_audio]).set_duration(duration)

    print(f"🎼 合成音声を '{temp_wav_path}' に保存中...")
    composite_audio.write_audiofile(temp_wav_path, fps=44100, codec='pcm_s16le')
    '''
    print("📜 字幕を生成中...")
    generate_subtitles(vocal_path, srt_path, offset_seconds, model_size)
    '''
    print("📖 字幕を読み込み中...")
    with open(srt_path, "r", encoding="utf-8") as f:
        subtitles = create_subtitle_clips(srt.parse(f.read()), offvocal_path, 1920, 1080)

    print("🎥 最終動画を生成中...")
    video = VideoFileClip(video_path).resize((1920, 1080)).set_duration(duration)
    create_final_video(video, subtitles, composite_audio, output_video)

    print("✅ 完了！出力:", output_video)

if __name__ == "__main__":
    main()
