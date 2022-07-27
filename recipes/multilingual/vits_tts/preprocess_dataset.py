def main():
    filepath = r'C:\Users\PiaoYang\Desktop\aidatatang_200zh\transcript\aidatatang_200_zh_transcript.txt'
    with open(filepath, encoding='UTF-8') as f:
        lines = f.readlines()
    with open(filepath.replace('.txt', '.csv'), 'w', encoding='UTF-8') as f:
        f.writelines(
            [line[:15] + '|' + line[15:].replace(' ', '', -1) for line in lines]
        )


if __name__ == '__main__':
    main()
