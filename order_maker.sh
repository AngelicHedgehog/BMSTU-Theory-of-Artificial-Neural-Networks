pandoc \
  --pdf-engine=xelatex \
  -V 'mainfont:Liberation Serif' \
  -V 'monofont:Liberation Mono' \
  "hw$1/readme.md" -o "hw$1/ДЗ№$1_ИУ9-72_ВиленскийСД.pdf"
