#!/bin/sh
git status
git add *  
git commit -m 'add some code from Mac'
git pull --rebase origin reedpan   #download data
git push origin reedpan           #upload data