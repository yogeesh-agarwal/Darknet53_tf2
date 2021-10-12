import os
import random
import pickle
from icrawler.builtin import BingImageCrawler

def download_images(query , num_images , download_dir):
    bing_crawler = BingImageCrawler(downloader_threads=4 , storage={'root_dir': download_dir})
    bing_crawler.crawl(keyword=query , filters=None , offset=0 , max_num=num_images)

if __name__ == "__main__":
    search_queries = ["dog" , "car" , "food" , "buildings" , "mountains", "cat" , "animal" , "laptop" , "planets" , "sun" , "water" , "sky"]
    download_dir = "../data/neg_samples/"
    os.makedirs(download_dir , exist_ok=True)
    for query in search_queries:
        curr_download_dir = os.path.join(download_dir , query)
        if os.path.exists(curr_download_dir):
            os.makedirs(curr_download_dir)

        download_images(query , 500 , curr_download_dir)

    image_names = []
    for folder_name in os.listdir(download_dir):
        folder_path = os.path.join(download_dir , folder_name)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path , img_name)
            image_names.append(img_path)

    random.shuffle(image_names)
    train_image_names = random.choices(image_names , k = 2500)
    test_image_names = random.choices(image_names , k = 1000)

    for i in range(len(train_image_names)):
        print(i , train_image_names[i])

    print("****************************************************")
    for i in range(len(test_image_names)):
        print(i , test_image_names[i])


    with open("../data/neg_samples.pickle" , "wb") as content:
        pickle.dump(train_image_names , content , protocol = pickle.HIGHEST_PROTOCOL)

    with open("../data/neg_samples_test.pickle" , "wb") as content:
        pickle.dump(test_image_names , content , protocol = pickle.HIGHEST_PROTOCOL)

    print("Neg_samples pickle file is generated.")
