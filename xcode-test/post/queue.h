/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef BLOCKING_QUEUE_H
#define BLOCKING_QUEUE_H

#include <condition_variable>
#include <list>
#include <locale>
#include <mutex>
#include <stdint.h>

// static const int DEFAULT_MAX_QUEUE_SIZE = 5; //3 8
static const int DEFAULT_MAX_QUEUE_SIZE = 3; //3 8 不能太大，要不会有高延迟（如防压手等延迟要求高）

template <typename T> class BlockingQueue {
    public:
    uint32_t max_size_;

    BlockingQueue(uint32_t maxSize = DEFAULT_MAX_QUEUE_SIZE) : max_size_(maxSize) {}

    ~BlockingQueue() {}

    int Pop(T &item)  //pop默认是阻塞的
    {
        std::unique_lock<std::mutex> lock(mutex_);

        while (queue_.empty()) {
            empty_cond_.wait(lock);
        }

        if (queue_.empty()) {
            //APP_ERR_QUEUE_EMPTY
            return 2;
        } else {
            item = queue_.front();
            queue_.pop_front();
        }

        full_cond_.notify_one();

        return 0;
    }

 int Pop(T& item, unsigned int timeOutMs)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        auto realTime = std::chrono::milliseconds(timeOutMs);

        // while (queue_.empty() && !is_stoped_) {
        //     empty_cond_.wait_for(lock, realTime);
        // }
        if (queue_.empty()) {  
            empty_cond_.wait_for(lock, realTime);//wait_for先解锁，通知或者超时了，会重新获取锁返回，不能一直while，GetSize()不是加锁的，有可能（消费和退出同时进来）退出线程清空队列了，消费队列在这阻塞回收不了
        }

        if (queue_.empty()) {
            //APP_ERR_QUEUE_EMPTY
            return 2;
        } else {
            item = queue_.front();
            queue_.pop_front();
        }

        full_cond_.notify_one();

        return 0;
    }

    int ReadHead(T &item,bool (*func)(T item,int *dumpPackCnt,bool *isfinish),int *dumpPackCnt,bool *isfinish)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        while (queue_.empty() ) {
            empty_cond_.wait(lock);
        }

        if (queue_.empty()) {
            //APP_ERR_QUEUE_EMPTY
            return 2;
        } else {
            item = queue_.front();
            if(func(item,dumpPackCnt,isfinish))
                queue_.erase(queue_.begin());
            // queue_.pop_front();
            // queue_.erase(queue_.begin());
        }
        full_cond_.notify_one();
        return 0;
    }
    

    int Push(const T& item, bool isWait = false)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        while (queue_.size() >= max_size_ && isWait) {
            full_cond_.wait(lock);
        }

        if (queue_.size() >= max_size_) {
            //APP_ERROR_QUEUE_FULL
            return 3;
        }
        queue_.push_back(item);

        empty_cond_.notify_one();

        return 0;
    }

    int IsFull()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size() >= max_size_;
    }


    int GetSize()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

    int IsEmpty()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void Clear()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.clear();
    }


    std::list<T> queue_;
    std::mutex mutex_;
    private:    
    std::condition_variable empty_cond_;
    std::condition_variable full_cond_;
};
#endif // __INC_BLOCKING_QUEUE_H__