#ifndef WRITE_PRIOTITY_LOCK
#define WRITE_PRIOTITY_LOCK

#include <mutex>
#include <condition_variable>
//读优先可用boost的读写锁

class write_priotity_lock
{
public:
    void read_lock()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mReadCV.wait(lock,[this](){ //return true go on
            return this->mWriteCnt == 0;
        });
        ++mReadCnt;
    }

    void write_lock()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        ++mWriteCnt;
        mWriteCV.wait(lock,[this](){
            return this->mWriteCnt <= 1 && this->mReadCnt == 0;
        });
        
    }

    void read_release()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        --mReadCnt;
        if(mReadCnt == 0 && mWriteCnt > 0)
        {
            mWriteCV.notify_one();
        }
    }

    void write_release()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        --mWriteCnt;
        if(mWriteCnt == 0)
        {
            mReadCV.notify_all();
        }
        else
        {
            mWriteCV.notify_one();
        }
    }

    write_priotity_lock()
    {
        mReadCnt = 0;
        mWriteCnt = 0;
    }


private:
    std::condition_variable mWriteCV;
    std::condition_variable mReadCV;
    std::int32_t mReadCnt = 0;
    std::int32_t mWriteCnt = 0;
    std::mutex mMutex;
};

class AUTO_LOCK
{
public:
    AUTO_LOCK(write_priotity_lock& lock,bool readOrWirte):mReadOrWirte(readOrWirte),mlock(lock)
    {
        if(mReadOrWirte)
            mlock.read_lock();
        else
            mlock.write_lock();
    }
    ~AUTO_LOCK()
    {
        if(mReadOrWirte)
            mlock.read_release();
        else
            mlock.write_release();
    }

private:
    bool mReadOrWirte = false;
    write_priotity_lock& mlock;
};

#endif