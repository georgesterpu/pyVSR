
with open('allVideos.txt', 'r') as f:
	contents = f.read().splitlines()
	
train = []
test = []	
s_list= ['s6_', 's8_', 's9_', 's15_', 's26_', 's30_',
         's34_', 's43_', 's44_', 's49_', 's51_', 's52_']
         
for line in contents:
	if any(s in line for s in s_list):
		test.append(line)
	else:
		train.append(line)
		
with open('train.txt', 'w') as f:
	for line in train:
		f.write(line + '\n')

with open('test.txt', 'w') as f:
	for line in test:
		f.write(line + '\n')
	
