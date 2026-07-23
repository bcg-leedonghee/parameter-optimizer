vm1_to_everywhere:
	#rsync -rzP --size-only /data/98_model admin@192.169.14.60:/data01/Documents # vm2
	#rsync -rzP --size-only /data/98_model admin@192.169.14.74:~/Documents # vm3
	rsync -rzP --size-only /data/98_model admin@192.169.14.75:~/Documents # vm4

vm2_to_everywhere:
	rsync -rzP --size-only /data01/Documents/98_model/notebooks admin@192.169.14.59:/data/98_model # vm1
	rsync -avzP /data01/Documents/98_model admin@192.169.14.74:~/Documents # vm3
	rsync -avzP /data01/Documents/98_model admin@192.169.14.75:~/Documents # vm4


vm3_to_everywhere:
	rsync -rzP --size-only /home/admin/Documents/98_model/notebooks admin@192.169.14.59:/data/98_model # vm1
	rsync -avzP /home/admin/Documents/98_model admin@192.169.14.60:/data01/Documents # vm2
	rsync -avzP /home/admin/Documents/98_model admin@192.169.14.75:~/Documents # vm4


vm4_to_everywhere:
	rsync -rzP --size-only /home/admin/Documents/98_model admin@192.169.14.59:/data # vm1
	rsync -avzP /home/admin/Documents/98_model admin@192.169.14.60:/data01/Documents # vm2
	rsync -avzP /home/admin/Documents/98_model admin@192.169.14.74:~/Documents # vm3


