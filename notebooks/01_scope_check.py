import glob, os
import soundfile as sf

base = "../data/slider/id_00"

normal = sorted(glob.glob(os.path.join(base, "normal", "*.wav")))
abnormal = sorted(glob.glob(os.path.join(base, "abnormal", "*.wav")))

print("Normal files  :", len(normal))
print("Abnormal files:", len(abnormal))
print("Anomaly ratio :", round(len(abnormal) / (len(normal) + len(abnormal)), 3))

# Inspect one file
info = sf.info(normal[0])
print("\nExample file:", os.path.basename(normal[0]))
print("Sample rate  :", info.samplerate)
print("Channels     :", info.channels)
print("Duration (s) :", round(info.duration, 2))

# Check that ALL files share the same format
bad = []
for f in normal + abnormal:
    i = sf.info(f)
    if i.samplerate != 16000 or i.channels != 8 or abs(i.duration - 10) > 0.1:
        bad.append((f, i.samplerate, i.channels, i.duration))
print("\nFiles with unexpected format:", len(bad))