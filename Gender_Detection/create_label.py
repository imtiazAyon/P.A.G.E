import csv

with open('feature.csv') as file:
	reader = csv.reader(file)
	with open('Fairlabel.csv') as file2:
		reader2 = csv.reader(file2)
		with open('final_label3.csv', 'w', newline='') as file3:
			writer = csv.writer(file3)
			for line in reader:
				file2.seek(0)
				for line2 in reader2:
					if line[0] == line2[0]:
						writer.writerow(line2)
						break
