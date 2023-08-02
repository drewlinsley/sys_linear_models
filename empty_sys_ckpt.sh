echo "You are deleting old ckpts. Are you sure you want to do this?"
read yes
if [ $yes == "yes" ]
then
    echo "Deleting"
    rm -rf /media/data/sys_ckpts/*
else
    echo "Exiting"
fi
