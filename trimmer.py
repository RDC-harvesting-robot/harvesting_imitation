import sys
import argparse
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

def trim_bag(input_bag_path, output_bag_path, start_time_ns, end_time_ns):
    # readerの設定
    try:
        reader = SequentialReader()
        reader_storage_options = StorageOptions(uri=input_bag_path, storage_id='sqlite3')
        reader_converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(reader_storage_options, reader_converter_options)
    except Exception as e:
        print(f"Error opening bag file {input_bag_path}: {e}")
        sys.exit(1)

    # writerの設定
    writer = SequentialWriter()
    writer_storage_options = StorageOptions(uri=output_bag_path, storage_id='sqlite3')
    writer_converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    writer.open(writer_storage_options, writer_converter_options)

    # トピック情報を取得
    topic_types = reader.get_all_topics_and_types()
    
    # 新しいbagファイルにトピック情報を登録
    for topic_metadata in topic_types:
        writer.create_topic(topic_metadata)

    # メッセージをフィルタリングして書き込み
    messages_written = 0
    
    # 最初のメッセージを読み込んでタイムスタンプを取得したreaderは使い終わったので、
    # 新しいreaderで最初からメッセージを読み直す
    reader_for_trimming = SequentialReader()
    reader_for_trimming.open(reader_storage_options, reader_converter_options)

    while reader_for_trimming.has_next():
        topic_name, data, timestamp = reader_for_trimming.read_next()
        if start_time_ns <= timestamp <= end_time_ns:
            writer.write(topic_name, data, timestamp)
            messages_written += 1
            
    print(f"Bag file trimmed successfully. Messages written: {messages_written}")
    print(f"New bag saved to: {output_bag_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trim a ROS 2 bag file by time.")
    parser.add_argument('input_bag', type=str, help='Path to the input bag file.')
    parser.add_argument('output_bag', type=str, help='Path to the output bag file.')
    parser.add_argument('--start', type=float, default=0.0, help='Start time in seconds from the beginning of the bag.')
    parser.add_argument('--end', type=float, help='End time in seconds from the beginning of the bag. If not specified, trims to the end.')

    args = parser.parse_args()

    # bagファイルの開始時刻を取得するために、一時的にreaderを使用
    temp_reader = SequentialReader()
    try:
        temp_reader.open(StorageOptions(uri=args.input_bag, storage_id='sqlite3'), ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    except Exception as e:
        print(f"Error opening bag file {args.input_bag}: {e}")
        sys.exit(1)

    # 最初のメッセージを読み込んでタイムスタンプを取得
    if temp_reader.has_next():
        first_timestamp_ns = temp_reader.read_next()[-1]
    else:
        print("Error: The bag file is empty.")
        sys.exit(1)
    
    start_ns = first_timestamp_ns + int(args.start * 1e9)
    end_ns = first_timestamp_ns + int(args.end * 1e9) if args.end is not None else float('inf')

    trim_bag(args.input_bag, args.output_bag, start_ns, end_ns)