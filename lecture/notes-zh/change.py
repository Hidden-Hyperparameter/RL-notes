import re,os

def change(in_file,out_file):
    lines = open(in_file).readlines()
    outlines = []
    for line in lines:
        line = line.strip('\n') # changed here
        if line == '':
            outlines.append('')
            continue
        if '$' in line:
            splited = line.split('$')
            out = []
            for i,item in enumerate(splited):
                if i % 2 == 1:
                    pre = '' if (splited[i-1] == '' or splited[i-1][-1] in '*~ ') else ' '
                    post = '' if (i+1 == len(splited) or splited[i+1]=='' or splited[i+1][0] in '*~ ') else ' '
                    out.append(pre + '$' + item + '$' + post)
                else:
                    out.append(item)
            outlines.append(''.join(out))
        else:
            outlines.append(line)
    open(out_file, 'w').write('\n'.join(outlines))

if __name__ == '__main__':
    mds = os.listdir('.')
    mds = [md for md in mds if md.endswith('.md')]
    print(mds)
    for md in mds:
        change(md,md)