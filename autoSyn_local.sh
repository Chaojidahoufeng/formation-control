#!/bin/sh
git status
git add *  
git commit -m 'add formation error/spring error, reduce the observation'
# git commit -m 'add some results from Server'
git pull --rebase origin yuzi   #domnload data
git push origin yuzi       #upload data
