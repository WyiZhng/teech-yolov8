import json, sys, pathlib
in_path, out_path = sys.argv[1], sys.argv[2]
with open(in_path,'r',encoding='utf-8') as f: coco=json.load(f)
keep = {0,1}                      # 原本的两类索引（若原始是 1/2，就写 {1,2}）
anns=[]
for a in coco['annotations']:
    if a.get('iscrowd',0): continue
    if a['category_id'] in keep:
        a['category_id'] = 0
        anns.append(a)
coco['annotations'] = anns
coco['categories'] = [{'id':0,'name':'caries'}]
pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
json.dump(coco, open(out_path,'w',encoding='utf-8'), ensure_ascii=False)
