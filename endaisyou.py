import numpy as np
from pydub import AudioSegment
from moviepy.editor import VideoClip, AudioFileClip
import scipy.io.wavfile as wavfile
import cv2
import os

# 音声ファイルを読み込む関数
def load_audio(file_path):
    # MP3の場合、WAVに変換して読み込む
    if file_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(file_path)
        file_path = file_path.replace('.mp3', '.wav')
        audio.export(file_path, format="wav")
    
    # WAVファイルをそのまま読み込む
    rate, data = wavfile.read(file_path)

    # ステレオの場合、モノラルに変換（左右の平均を取る）
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    return rate, data.astype(np.int16)  # データ型を適切に変換

# 円のサイズを音に基づいて変化させる関数
def make_frame(t, audio_data, rate, max_radius=200):
    sample_idx = int(t * rate)
    sample_idx = min(sample_idx, len(audio_data) - 1)
    sample = audio_data[sample_idx]

    # 💡 最小半径と変動量（揺れ幅）を設定
    min_radius = 80           # 元の大きさ（基準サイズ）
    dynamic_range = 200        # 揺れる大きさの幅

    # 変動する半径の計算
    dynamic = dynamic_range * (abs(sample) / 32768.0)
    radius = int(np.clip(min_radius + dynamic, 0, max_radius))

    # 白背景
    #frame = np.ones((500, 500, 3), dtype=np.uint8) * 255
    frame = np.ones((500, 500, 3), dtype=np.uint8) * 255

    center = (250, 250)
    color = (173, 216, 230)  # 水色 (OpenCVはBGR)

    cv2.circle(frame, center, radius, color, -1)
    
    return frame

    
    return frame

# 動画を作成する関数
def create_audio_circle_video(audio_file, output_video='output.mp4', fps=30):
    # 音声をロード
    rate, data = load_audio(audio_file)
    
    # 音声の長さをフレーム数に変換
    duration = len(data) / rate
    
    # フレーム描画関数
    def make_frame_with_audio(t):
        return make_frame(t, data, rate)
    
    # 動画の作成
    clip = VideoClip(make_frame_with_audio, duration=duration)
    
    # 音声を動画に追加
    audio_clip = AudioFileClip(audio_file)
    video = clip.set_audio(audio_clip)
    
    # 動画を書き出し
    video.write_videofile(output_video, codec='libx264', fps=fps)

# 実行部分
audio_file = 'yoidore.mp3'  # ここに使用するMP3/WAVファイルのパスを指定
create_audio_circle_video(audio_file, 'output_video_odiospe2.mp4')
