import os

folder_path = "/mnt/sda/minkyukim/sam_dataset/coco_npy_test_dataset_1024image/gts"

files_to_delete = [
    "100723_14.npy", "105912_13.npy", "107851_14.npy", "108503_14.npy", "114907_14.npy",
    "117492_14.npy", "12120_14.npy", "122962_14.npy", "125572_14.npy", "126137_13.npy",
    "12639_14.npy", "12670_14.npy", "133244_14.npy", "133969_14.npy", "134322_14.npy",
    "136715_14.npy", "138639_14.npy", "139099_15.npy", "141671_14.npy", "142238_14.npy",
    "142324_14.npy", "143068_13.npy", "143572_14.npy", "143961_16.npy", "14439_14.npy",
    "145020_14.npy", "148508_14.npy", "148999_14.npy", "153669_14.npy", "154004_14.npy",
    "156071_14.npy", "158548_14.npy", "161978_20.npy", "16228_14.npy", "163057_14.npy",
    "163314_14.npy", "165681_12.npy", "166747_14.npy", "169169_14.npy", "173057_14.npy",
    "179487_13.npy", "17959_14.npy", "181542_14.npy", "182923_14.npy", "183391_14.npy",
    "18380_14.npy", "184324_14.npy", "184611_14.npy", "188465_14.npy", "189775_14.npy",
    "190676_14.npy", "190923_14.npy", "191845_14.npy", "193181_14.npy", "193245_14.npy",
    "199236_14.npy", "201418_14.npy", "203931_14.npy", "204186_14.npy", "207844_14.npy",
    "208363_14.npy", "209222_14.npy", "210394_14.npy", "211674_14.npy", "212800_14.npy",
    "214539_14.npy", "217285_14.npy", "219271_14.npy", "222559_12.npy", "224807_14.npy",
    "226802_14.npy", "227399_14.npy", "229849_14.npy", "2299_14.npy", "231508_14.npy",
    "236426_14.npy", "24021_14.npy", "24243_14.npy", "250282_14.npy", "254814_14.npy",
    "256192_14.npy", "259571_14.npy", "259597_14.npy", "259690_14.npy", "261097_14.npy",
    "266892_14.npy", "26690_15.npy", "266981_14.npy", "272148_13.npy", "277005_14.npy",
    "279927_14.npy", "282037_14.npy", "282298_14.npy", "283785_14.npy", "284445_14.npy",
    "288685_15.npy", "290248_14.npy", "293200_14.npy", "296649_13.npy", "299553_14.npy",
    "303566_14.npy", "303713_14.npy", "304404_14.npy", "305317_14.npy", "309391_14.npy",
    "312237_13.npy", "325031_14.npy", "328430_14.npy", "326248_13.npy", "329323_14.npy",
    "331280_14.npy", "332351_14.npy", "336658_14.npy", "338219_14.npy", "338304_14.npy",
    "343561_14.npy", "345027_14.npy", "350122_14.npy", "356387_14.npy", "361103_14.npy",
    "361730_14.npy", "361919_14.npy", "363188_14.npy", "364884_14.npy", "36494_14.npy",
    "370486_14.npy", "373382_14.npy", "374545_14.npy", "376112_14.npy", "376322_14.npy",
    "37689_14.npy", "376900_14.npy", "377723_14.npy", "378673_15.npy", "388846_14.npy",
    "390555_14.npy", "39405_14.npy", "39484_14.npy", "410496_14.npy", "410650_14.npy",
    "414673_14.npy", "402615_14.npy", "406611_13.npy", "412894_14.npy", "397303_14.npy",
    "411530_14.npy", "421060_14.npy", "427256_14.npy", "427997_14.npy", "430073_13.npy",
    "432898_14.npy", "433515_14.npy", "438955_14.npy", "439180_14.npy", "439994_14.npy",
    "441553_14.npy", "448263_14.npy", "450399_14.npy", "455624_14.npy", "463522_15.npy",
    "463542_14.npy", "463730_14.npy", "470924_14.npy", "474028_14.npy", "476770_14.npy",
    "478862_14.npy", "481390_14.npy", "484351_14.npy", "486104_14.npy", "489842_14.npy",
    "490936_14.npy", "493905_14.npy", "49759_14.npy", "5001_14.npy", "500478_14.npy",
    "504000_14.npy", "507037_14.npy", "509014_14.npy", "520659_14.npy", "520707_14.npy",
    "521259_14.npy", "524850_14.npy", "531135_14.npy", "540414_14.npy", "541123_14.npy",
    "542127_14.npy", "568439_14.npy", "572620_16.npy", "57597_14.npy", "57760_17.npy",
    "559842_14.npy", "561958_12.npy", "562818_14.npy", "563648_22.npy", "57150_13.npy",
    "558114_14.npy", "5586_14.npy", "559348_14.npy", "570688_14.npy", "570756_14.npy",
    "546823_13.npy", "551820_14.npy", "59044_14.npy", "59635_14.npy", "60507_14.npy",
    "60886_14.npy", "65288_14.npy", "65455_14.npy", "65798_14.npy", "72795_13.npy",
    "74092_14.npy", "7511_14.npy", "76468_14.npy", "77460_14.npy", "78565_15.npy",
    "78748_14.npy", "79969_14.npy", "84031_14.npy", "84270_14.npy", "85682_14.npy",
    "87038_15.npy", "94157_14.npy", "95862_14.npy", "97988_13.npy", "98287_14.npy",
    "98853_14.npy", "99114_14.npy"
]

for file_name in files_to_delete:
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_name} 삭제 완료")
    else:
        print(f"{file_name} 파일을 찾을 수 없습니다")

print("모든 파일 삭제 완료")
