ALPHABET = u"耳夢建早样事波球成外畫局嘉主但阳道諸乾或路故長的灵勿御岸去部杜四電高苦番方切威聲曰聚班端數始瘦床水遺魚烏武重沉轉清傳觉油火手鄉地此底鳥收貴偏計起用當号弄件亦深運是包代妙艮良庭亮英沙還竟语碎其孝华幽斷身永和表機国離官着連背樓名多微父刀开首化絕業體夏峰易餘無当请洞零白许利野入晨美亭蘇東繁别见土兴葉尊思走似理昔管源要吾相為次期船亡步帶氏阿拉进今果姐逐場窮罗君近色驚强万字命没力客盛將处特雨祖久暗蓮轻遠范位睡侯群求可食降龍樹肉米裡飛些病存射季途後把不梁眼雲条親回末商系屋智寒時散這關乃浩待家卷赤殺布雷蘭里虛雪華甚春頭終勢著勝村黃眠臨宏府累矣影车吹宝说基孫盤女段窗候旅爱傾比齊湖心圖服遊先寄初足月種各紅能付除甘題書声露桂消聞界草往芳穿調东算登獨右会洗玉式遇言休六景慶羽乘應識出破澤鳴乎奈辰很等们复德孤放沈一節論真海柳千夂宜叫瑞蛋彩个看日校傷車說惠修血无性子精味垂集克指害豐靜義便政短報唐九羅也復圓吃軍合張考康列敬台刻皇語師臺悠送业社塞曾號角漢溪梅悲加活陰套住發喜胡宇百魂俊科欠正限歸健石又問配夫平豪觀毛任夕施森昭晴已周河淡徐乐目下逆那过既醉極伯排菜汝田将中室人查止楚仁守使世未誰到承寂友干惡钱氣童昌哀陽忽向二最口戰泉你而以受半急分博榮珍灰息敢院想在緣狂伊失黎滿第泰鬼征忠得賢章銀左孟空八暖渡樂隔來禮臣謝金喝輕治才就阮志象古男凡疑丹量須李吳衣飲天告奇器間为有张迎與五常川如上難只開像過盡封立宗兒支戈小恨爾細文完則非知顧星懷带兵遍度宿酒愁亲進儿郎冠尚佛含好風还坤对云反般怨吟卒做教牙王從萬佳惟共何舉竹经希聽江許我死感北浮采这大引玄夜者國忍快再格师羊薄学演南幸都居场山照交虎順雄流气西折笑寶谢神愛然異塵发浪望太少及紫元同煙實辛朝并暴他若弟龙杯余信恩从園總秋馬桑明午推差留体造翠彼辭慧关牛猶形池跟宮林聖料自覺面青丁之本予井魔延通壁鹿達直持年情与动机柔意伏筆奉徒司市桃戴門兮鐘骨改皮尔堂申实依己陳暮解翼香興片母符斯聊哥巴燕曲七舊低马蒼沒琴超且雅陈念至越甲詩什朱福省於几点忘記士静所令游會风坐民冷秀姓動行枝新忙鼓停欲藏私法茶店更城靈长物生木禾願假秦昏淫見每髮全示經后蕭餐晚雙益刺吉寸皆黑別了逢被老強程原歌斗欣冰倒移接来却取功舍时松雞富黄即迷定舞莫磨尋红座谷书音舟舒變离蒙寺霜碧咸打州邊幾么间前族提翻結招整致橋歲冬麻典席头仙條具錄花借制数保由公光必隨根抱昨呼容處工映三落帝壽品祥因善于京难素作兩安麗學珠亂橫對房婦十尾勤楊寧"

MAX_STR_LEN = 12 # max length of input labels
MIN_STR_LEN = 6
NUM_OF_CHARACTERS = len(ALPHABET) + 1 # +1 for ctc pseudo blank
NUM_OF_TIMESTEPS = 16 # max length of predicted labels

SAVE_TEXT_IMAGE_TO_DISK = False
FONT_SIZE = 20
FONT_SIZE_MIN = 18
FONT_SIZE_MAX = 18
IMG_WIDTH = 256
IMG_HEIGHT = 64

DATASET_DIR = 'data/'
TRAIN_DIR = DATASET_DIR + 'train/'
VALID_DIR = DATASET_DIR + 'valid/'
TEST_DIR = DATASET_DIR + 'test/'
TRAIN_IMG_DIR = TRAIN_DIR + 'images/'
VALID_IMG_DIR = VALID_DIR + 'images/'
TEST_IMG_DIR = TEST_DIR + 'images/'

LABEL_FILE_NAME = 'labels.txt'
TRAIN_LABEL_FILE = TRAIN_DIR + LABEL_FILE_NAME
VALID_LABEL_FILE = VALID_DIR + LABEL_FILE_NAME
TEST_LABEL_FILE = TEST_DIR + LABEL_FILE_NAME

DATASET_FILE_NAME = 'dataset.csv'
TRAIN_DATASET_FILE = TRAIN_DIR + DATASET_FILE_NAME
VALID_DATASET_FILE = VALID_DIR + DATASET_FILE_NAME
TEST_DATASET_FILE = TEST_DIR + DATASET_FILE_NAME

FONT_LIST = 'fonts/fonts.txt'
NOMCORPUS_FILE = 'fonts/base/seed.txt'
ALPHABETS_FILE = 'fonts/alphabet.txt'
CHECKPOINT_DIR = 'checkpoints'

TRAIN_SIZE = 30000
VALID_SIZE = 3000
CHUNK_SIZE = 5000