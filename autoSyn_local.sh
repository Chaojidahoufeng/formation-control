#!/bin/sh
git status
git add *  
git commit -m 'add some code from Mac'
# git commit -m 'add some results from Server'
git pull --rebase origin yuzi   #domnload data
git push        #upload data