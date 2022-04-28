#!/bin/bash
#find -L $1 -type f \( -iname "*.ape" -o -name "*.wav" -o -iname "*flac" \) -exec ffmpeg -i {} -ar 22050 "$2"/{}.wav \; -exec rm {} \;
#find -L $1 -type f \( -iname "*.ape" -o -name "*.wav" -o -iname "*flac" \) -exec sh -c 'echo "$2"/`basename "{}"`.wav' \; 
#find -L $1 -type f \( -iname "*.ape" -o -name "*.wav" -o -iname "*flac" \) -exec sh -c 'ffmpeg -i "{}" -ar 22050 "$2""`basename "{}"`".wav' \; 

#find -L "$1" -type f \( -iname "*.ape" -o -name "*.wav" -o -iname "*flac" \) | cat -n | while read n f; do ffmpeg  -y -i "$f" -ar 22050 "$2""$n".wav </dev/null ; done
find -L "$1" -type f \( -iname "*.ape" -o -iname "*.wav"  -o -iname "*.mp3" -o -iname "*flac" \) | cat -n | while read n f; do mkdir -p "$2/`dirname $f`"  ;  ! test -f "$2"/"$f".wav && sleep 1.2 ; ffmpeg  -n -i "$f" -ar 44100 "$2"/"$f".wav </dev/null & done
