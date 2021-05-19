#!/bin/sh
git status
git add *  
git commit -m '5-19: add spring model'
# git commit -m 'add some results from Server'
git pull --rebase origin yuzi   #domnload data
git push origin yuzi       #upload data
