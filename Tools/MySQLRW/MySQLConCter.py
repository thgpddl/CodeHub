import pymysql


class MySQLAbstract:
    def __init__(self):
        pass

    def _connect(self, host, user, password, database, table):
        self.table = table
        try:
            self.db = pymysql.connect(host=host, user=user, password=password, database=database, charset='utf8')
            self.cursor = self.db.cursor()  # 游标
            # 先执行，再获取结果
            self.cursor.execute("SELECT VERSION()")
            version = self.cursor.fetchone()  # 使用 fetchone() 方法获取一条数据
            print(f"Database version：{version}")
            print("数据库连接成功")
        except:
            print("数据库连接失败")

    def _isTableExists(self):
        sql = f"SELECT COUNT(*) FROM information_schema.TABLES WHERE table_name ='{self.table}';"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()  # 获取所有数据
        return results

    def getQueryData(self):
        sql = f"SELECT * FROM {self.table}"
        try:
            # 执行sql语句
            self.cursor.execute(sql)
            results = self.cursor.fetchall()  # 获取所有数据
            return results
        except:
            print("Error: unable to fetch data")
            return None

    def close(self):
        self.db.close()


class MySQLForEmotion(MySQLAbstract):
    def __init__(self, host, user, password, database, table):
        super().__init__()
        self._connect(host, user, password, database, table)

    # def insert_data_batch(self, emotion_data):
    #     """
    #
    #     :param data:
    #             data = {
    #                 "time": '2022-06-21-20-00-20',
    #                 "data": {
    #                     "positive": str,
    #                     "negative": str,
    #                     "neutral": str,
    #                 }
    #             }
    #     :return: None
    #     """
    #     keys = list(emotion_data['data'].keys())
    #     sql = f"""INSERT INTO {self.table}(time, value, type) VALUES (%s,%s,%s)"""
    #     val = []
    #     # for i in range(3):
    #     for key in keys:
    #         time = emotion_data["time"]
    #         value = emotion_data["data"][key]
    #         type = key
    #         val.append((time, value, type))
    #
    #     try:
    #         # 执行sql语句
    #         self.cursor.executemany(sql, val)
    #         # 提交到数据库执行
    #         self.db.commit()
    #     except MySQLdb._exceptions.OperationalError as e:
    #         # Rollback in case there is any error
    #         print("EmotionError: unable to insert data")
    #         print(e)
    #         self.db.rollback()

    def insert_emotion_data_batch(self, emotion_data):
        """

        :param data:
                emotion_data = {
                                "time": time_data,
                                "confidence": getEmotionData(), 列表，元素为字符串
                                "diff": getEmotionData(), 列表，元素为字符串
                            }
        :return: None
        """
        sql = f"""INSERT INTO {self.table}(time, confidence, diff, serial) VALUES (%s,%s,%s,%s)"""
        val = []
        time = emotion_data["time"]
        for i in range(emotion_data["confidence"].__len__()):
            confidence = emotion_data["confidence"][i]
            diff = emotion_data["diff"][i]
            val.append((time, confidence, diff,str(i)))

        try:
            # 执行sql语句
            self.cursor.executemany(sql, val)
            # 提交到数据库执行
            self.db.commit()
        except pymysql._exceptions.OperationalError as e:
            # Rollback in case there is any error
            print("EmotionError: unable to insert data")
            print(e)
            self.db.rollback()

    def insert_pbm_data(self, heartbeat_data):
        """

        :param img_data:
                        heartbeat={
                                "time": time_data,
                                "pbm":random.randint(0,100)
                            }
        :return:
        """
        sql = f"""INSERT INTO {self.table}(time,pbm) VALUES ('{heartbeat_data["time"]}',{heartbeat_data["pbm"]});"""

        try:
            # 执行sql语句
            self.cursor.execute(sql)
            # 提交到数据库执行
            self.db.commit()
        except pymysql._exceptions.OperationalError as e:
            # Rollback in case there is any error
            print("ImgsError: unable to insert data")
            print(e)
            self.db.rollback()


class MySQLForImgs(MySQLAbstract):
    def __init__(self, host, user, password, database, table):
        super().__init__()
        self._connect(host, user, password, database, table)
        self._toEmptyTable()

    def _toEmptyTable(self):
        sql = "delete from imgs"
        try:
            # 执行sql语句
            self.cursor.execute(sql)
            # 提交到数据库执行
            self.db.commit()
            print("表已清空")
        except:
            # Rollback in case there is any error
            print("Error: unable to opti data")
            self.db.rollback()

    def _itemopti(self):
        if self.getQueryData().__len__() > 50:
            # 删除
            sql = "delete A from imgs A join (select time from imgs ORDER BY time limit 5) B on A.time<=B.time;"
            try:
                # 执行sql语句
                self.cursor.execute(sql)
                # 提交到数据库执行
                self.db.commit()
                print("清除成功")
            except:
                # Rollback in case there is any error
                print("Error: unable to opti data")
                self.db.rollback()

    def insert_imgs(self, img_data):
        """

        :param img_data:
                        img_data = {
                                    "base64": base64c.encode(ndarray=frame),
                                    "time": '2022-06-21-20-00-20',
                                    }
        :return:
        """
        sql = f"""INSERT INTO {self.table}(base64,time) VALUES ('{img_data["base64"]}','{img_data["time"]}');"""

        try:
            # 执行sql语句
            self.cursor.execute(sql)
            # 提交到数据库执行
            self.db.commit()
        except pymysql._exceptions.OperationalError as e:
            # Rollback in case there is any error
            print("ImgsError: unable to insert data")
            print(e)
            self.db.rollback()
        self._itemopti()
