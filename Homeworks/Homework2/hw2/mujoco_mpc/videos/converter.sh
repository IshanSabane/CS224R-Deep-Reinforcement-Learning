
for f in *.avi; do
    if [ -f "${f%.*}.mp4" ]; then
        echo "${f%.*}.mp4 already exists"
    else
        ffmpeg -i "$f" "${f%.*}.mp4"
    fi
done
