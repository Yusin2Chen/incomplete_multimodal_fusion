import json

annotation_file = '/media/yusin/Elements/DFC2023/track2/track2-ori.json'

with open(annotation_file, "r+") as fcc_file:
    fcc_data = json.load(fcc_file)
    print(fcc_data['annotations'][0])
    del_list = []
    for i in range(len(fcc_data['annotations'])):
        #print(fcc_data['annotations'][i])
        if not isinstance (fcc_data['annotations'][i]['segmentation'], list):
            print(fcc_data['annotations'][i]['segmentation'])
            del_list.append(i)

    del_list = del_list[::-1]
    for i in del_list:
        print(i)
        del fcc_data['annotations'][i]
        #try:
        #    if 'counts' in fcc_data['annotations'][i]['segmentation']:
        #        #print(fcc_data['annotations'][i]['segmentation'])
        #        del fcc_data['annotations'][i]
        #except:
        #    pass

with open('/media/yusin/Elements/DFC2023/track2/track2-2.json', "w", encoding="UTF-8") as e:
    json.dump(fcc_data, e)
