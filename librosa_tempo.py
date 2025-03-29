#ライブラリインポート
import librosa                                                  #音声解析用のライブラリ

#変数定義
sampling_rate = 44100                                           #サンプリングレート
filename = '/Users/konatama/workspace/yoidore_offvocal.wav'                             #解析する音声ファイルのパス

#演算
y, sr = librosa.load(path=filename, sr = sampling_rate)         #音声ファイルを1次元のNumPy浮動小数点配列（変数y）に格納
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)        #テンポを算出

#結果表示
#print('Estimated tempo: {:.2f} beats per minute'.format(tempo)) #結果を表示
print(tempo)
