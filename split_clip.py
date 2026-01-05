import os
from moviepy import VideoFileClip

base_dir = r""

for cate_id in range(1, 14):
    cate_name = f"{cate_id:02d}"
    cate_path = os.path.join(base_dir, cate_name)
    out_2s = os.path.join(base_dir, f"{cate_name}_2s")
    out_4s = os.path.join(base_dir, f"{cate_name}_4s")
    out_6s = os.path.join(base_dir, f"{cate_name}_6s")
    os.makedirs(out_2s, exist_ok=True)
    os.makedirs(out_4s, exist_ok=True)
    os.makedirs(out_6s, exist_ok=True)

    for file in os.listdir(cate_path):
        if not file.endswith(".mp4"):
            continue

        file_path = os.path.join(cate_path, file)
        clip = VideoFileClip(file_path)
        base_filename = os.path.splitext(file)[0]

        if base_filename.startswith(f"{cate_name}_"):
            base_filename = base_filename[len(cate_name) + 1:]  

        for sec, out_dir in [(2, out_2s), (4, out_4s), (6, out_6s)]:
            if clip.duration >= sec:
                new_name = f"{base_filename}_{sec}s.mp4" 
                output_path = os.path.join(out_dir, new_name)
                new_clip = clip.subclipped(0, sec)
                new_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    logger=None
                )

                new_clip.close()

        clip.close()

print("Done.")
