#include "face_db.h"
#include "common/http/crow_all.h"
using idx_t = faiss::idx_t;
using namespace std;

FaceDb::FaceDb(int d):dimension(d)
{
    index = NULL;
    isInit = false;
}

FaceDb::~FaceDb()
{
    if(index)
        delete index;
}

int FaceDb::getSize()
{
    if(index!=nullptr)
        return index->ntotal;
    else
        return 0;
}

int FaceDb::init(vector<float> input)
{
    if(dimension < 0)
        return -1;
    index = new faiss::IndexFlatL2(dimension);
    printf("[face db] is_trained = %s\n", index->is_trained ? "true" : "false");
    printf("[face db] input = %d\n", input.size()/dimension);

    faiss::fvec_renorm_L2(dimension,input.size()/dimension,input.data());

  
    index->add(input.size()/dimension, input.data()); // add vectors to the index


    idIndex = input.size()/dimension;
    printf("[face db] ntotal = %zd\n", index->ntotal);
    isInit = true;
    return 0;
}

int FaceDb::init2(std::string indexDbPath)
{
    if(dimension < 0)
        return -1;
    index = new faiss::IndexFlatL2(dimension);

    if(access(indexDbPath.c_str(),F_OK) == 0)
    {
        index = (faiss::IndexFlatL2*)faiss::read_index("face_detection/facedb.bin"); //陌生人才有特征库，数据库有图片重新载入，所以没有特征库
        cout<<"[INFO] Read From Strangers.bin\n";
    }
    isInit = true;
    idIndex = 99;
    return 0;
}

int FaceDb::add(int num,vector<float> input)
{
    if(!isInit || input.size()<=0)
        return -1;
    
    faiss::fvec_renorm_L2(dimension,num,input.data());

    index->add(num, input.data());
    idIndex+=num;
    return 0;
}

int FaceDb::del(int n, idx_t *arr)
{
    faiss::IDSelectorArray tmp(n,arr);
    index->remove_ids(tmp);
    return 0;
}

int FaceDb::search2(int ik,int nq,vector<float> ixq,vector<idx_t> &iI,vector<float> &iD)
{
    // printf("%d-%d-%d\n",!isInit,ixq.size(),idIndex);
    if(!isInit || ixq.size()<=0 || idIndex<=0)
        return -1;

    int k = ik;
    
    if(idIndex < 2)
        k = 1;


    idx_t* I = new idx_t[k * nq];
    float* D = new float[k * nq];

    // float* xq = new float[dimension * nq;

    faiss::fvec_renorm_L2(dimension,nq,ixq.data());

    index->search(nq, ixq.data(), k, D, I);

    for(int i = 0; i < nq; i++)
    {
        for(int j = 0; j < k; j++)
        {
            iD.push_back(D[i*k + j]);
            iI.push_back(I[i*k + j]);
        }
    }

    delete[] I;
    delete[] D;

    return 0;
}

int FaceDb::search(int ik,int nq,vector<float> ixq,vector<idx_t> &iI,vector<float> &iD)
{
    // printf("%d-%d-%d\n",!isInit,ixq.size(),idIndex);
    if(!isInit || ixq.size()<=0 || idIndex<=0)
        return -1;

    int k = ik;
    
    if(idIndex < 2)
        k = 1;

    idx_t* I = new idx_t[k * nq];
    float* D = new float[k * nq];

    // float* xq = new float[dimension * nq;

    faiss::fvec_renorm_L2(dimension,nq,ixq.data());

    index->search(nq, ixq.data(), k, D, I);

    for(int i = 0; i < nq; i++)
    {
        bool isSearchDelId = true;
        for(int j = 0; j < k; j++)
        {
            iD.push_back(D[i*k + j]);
            iI.push_back(I[i*k + j]);
            break;
        }
    }
    
    delete[] I;
    delete[] D;

    return 0;
}

int FaceDb::searchforTool(int ik,int nq,vector<float> ixq,vector<idx_t> &iI,vector<float> &iD)
{
    // printf("%d-%d-%d\n",!isInit,ixq.size(),idIndex);
    if(!isInit || ixq.size()<=0 || idIndex<=0)
        return -1;

    int k = ik;
    
    if(idIndex < 2)
        k = 1;


    idx_t* I = new idx_t[k * nq];
    float* D = new float[k * nq];

    // float* xq = new float[dimension * nq;

    faiss::fvec_renorm_L2(dimension,nq,ixq.data());

    index->search(nq, ixq.data(), k, D, I);

    for(int i = 0; i < nq; i++)
    {
        
        for(int j = 0; j < k; j++)
        {
           
            iD.push_back(D[i*k + j]);
            iI.push_back(I[i*k + j]);
            
        }
    }
    delete[] I;
    delete[] D;

    return 0;
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------
void FaceServer::save_mind_data_thread()
{
    while(1)
    {
        int flag = 0;
        if(saveFlagQue->Pop(flag) == 0)
        {
            // while(saveFlagQue->Pop(flag,100) == 0)
            // {
            //     usleep(10*1000);
            // }
            std::map<idx_t,std::string> tmpMap = spIdFileNameMap;
            // std::map<idx_t,std::string> tmpMap;
            // {
            //     AUTO_LOCK wlock(dbLock,0);
            //     std::map<idx_t,std::string> tmpMap = spIdFileNameMap;
            // }
            idx_t idIndexTmp = idIndex;
            Json::Value saveData;
            Json::Value jsonarr;
            saveData["newIndex"] = (int)idIndex;
            
            for(auto it = tmpMap.begin(); it!= tmpMap.end();it++)
            {
                Json::Value tmp;
                tmp["fn"] = it->second;
                tmp["id"] = (int)it->first;
                jsonarr.append(tmp);
            }
            saveData["data"] = jsonarr;
            std::string saveStr = saveData.toStyledString();
            FILE * savePtr  = fopen("face_detection/FACE.bin","wb+");
            fwrite(saveStr.c_str(),1,saveStr.size(),savePtr);
            fclose(savePtr);
            std::cout<<"[INFO] save json in host!" <<std::endl;    
            if(_reload)
            {
                AUTO_LOCK wlock(dbLock,0);
                std::cout<<"[WARN] faceServer set req, need Reload!"<<std::endl;
                sleep(1);
                _Exit(4);
            }
                

        }
        usleep(300*1000);
    }
}

FaceServer::FaceServer(string inPath):_reload(false)
{
    mImgsPath = inPath;
    idIndex = 0;
    mStatus = false;
    int ret = 0;
    facedb = new FaceDb(512);
    

    
//    facetool.init("faceserver");
	faceRec::instance().init("");
    

    
    DIR *dir;
    struct dirent *ent;
    vector<string> imgNames;

    FILE *fPtr = fopen("face_detection/FACE.bin","rb");
    if(fPtr!=NULL)
    {
        string fileJson;
        char a;
        while(fread(&a,sizeof(char),1,fPtr) == 1)
        {
            fileJson.push_back(a);
        }
        Json::Reader reader;
        Json::Value value;
        reader.parse(fileJson,value);

        if(!value.isObject())
        {
            cout<<"[ERROR] Read FACE.bin fail!\n";
        }
        else
        {
            idIndex = value["newIndex"].asInt();
            for(int i = 0; i < value["data"].size(); i++)
            {
                imgNames.push_back(value["data"][i]["fn"].asString());
                spIdFileNameMap[value["data"][i]["id"].asInt()] = value["data"][i]["fn"].asString();
                fileNameSpIdMap[value["data"][i]["fn"].asString()] = value["data"][i]["id"].asInt();
            }
        }

        fclose(fPtr);
    }

    vector<float> dbData;
    for(int i = 0; i < imgNames.size(); i++)
    {
        cv::Mat img = cv::imread(mImgsPath+"/"+imgNames[i]);
        std::cout << mImgsPath+"/"+imgNames[i] << std::endl;
        if(img.empty())
        {
            cout<<"[ERROR] img is empty"<<endl;
        }

        std::vector<Yolov5Face_BoxStruct> face_result;    
        int ret1 =facetool.infer(img,face_result,dbData);   
        if(!ret1)
            continue;

        spIdVec.push_back(fileNameSpIdMap[imgNames[i]]);
    }

    ret = facedb->init(dbData);
    if(ret)
    {
        cout<<"[ERROR] db init imgs fail!"<<endl;
        // return ;
    }
    
    mStatus = true;

    saveFlagQue = new BlockingQueue<int>(9999);
    
    th_ptr2 = new std::thread(&FaceServer::save_mind_data_thread,this);
    th_ptr = new std::thread(&FaceServer::server_th,this);

}

FaceServer::~FaceServer()
{
    mStatus = false;
    delete facedb;
    facedb = NULL;    

}


int FaceServer::search(std::vector<faceInfo> &ans,std::vector<Yolov5Face_BoxStruct> &face_result,std::vector<int> &delNums,std::vector<float> &input,int querySize, int kSize)
{
    vector<float> D;
    vector<idx_t> I;
    I.resize(0);
    D.resize(0);


    if(querySize)
        facedb->search(kSize,querySize,input,I,D);

    ans.resize(0);


    int queryId = 0;
    
    for(int j = 0; j<face_result.size(); j++)
    {
        faceInfo tmp;
        tmp.x = face_result[j].x1;
        tmp.y = face_result[j].y1;
        tmp.w = face_result[j].x2 -  face_result[j].x1;
        tmp.h = face_result[j].y2 -  face_result[j].y1;
        tmp.score = face_result[j].score;

        if(std::count(delNums.begin(),delNums.end(),j))
        {
            tmp.conf = 0;
            tmp.id = -1;
            tmp.fileName = "";
        }
        else
        {
           if(facedb->index->ntotal <= 0)//数据库没人
            {
                tmp.conf = 0;
                tmp.id = -2;
                tmp.fileName = "";
                
            }
            else//数据库有人
            {
                tmp.conf = D[queryId];
                tmp.id = I[queryId];
                tmp.fileName = spIdFileNameMap[spIdVec[I[queryId]]];
                tmp.spId = spIdVec[I[queryId]];
            }
            tmp.score = face_result[j].score;
            queryId++;
        }

        ans.push_back(tmp);
    }
    return 0;
}



static string getGoodResp(int code,string msg)
{
    cout<<"msg:"<<msg<<endl;
    Json::Value ans;
    ans["code"] = code;
    ans["msg"] = msg;

    return ans.toStyledString();
}
static string getGoodIdResp(int code,string msg,long int id)
{
    cout<<"msg:"<<msg<<endl;
    Json::Value ans;
    ans["code"] = code;
    ans["msg"] = msg;
    ans["spId"] = (int)id;

    return ans.toStyledString();
}

extern float MaxConfLimit;
string FaceServer::getSearchResp(vector<idx_t> I,vector<float> D)
{
    Json::Value ans;
    Json::Value confs;
    Json::Value ids;
    Json::Value fileNames;
    ans["code"] = 200;
    for(auto item:D)
    {
        if(item>MaxConfLimit)
            continue;
        confs.append(item);
    }
    std::cout<<"search database cout:"<<D.size()<<endl;
    for(int i = 0; i < confs.size();i++)
    {
        ids.append((int)spIdVec[I[i]]);
        auto fileName = spIdFileNameMap[spIdVec[I[i]]];
        std::cout<<fileName<<","<<D[i]<<",spid:"<<(int)spIdVec[I[i]]<<endl;
        fileNames.append(fileName);
    }

    ans["msg"] = "success";
    ans["confs"] = confs;
    ans["ids"] = ids;
    ans["fileNames"] = fileNames;

    return ans.toStyledString();
}

void FaceServer::server_th()
{
    crow::SimpleApp app;
    
    CROW_ROUTE(app, "/cv/face_recognition/add/")
    .methods("POST"_method)
    ([&](const crow::request& req){
        // if(mStatus)
        //     return crow::response{getGoodResp(500,"db not init")};
        
        try{
            AUTO_LOCK wlock(dbLock,0);
            int ret;
            Json::Reader reader;
            Json::Value value;
            reader.parse(req.body,value);
            
            string imgName = value["imgName"].asString();
            
            string name;

            size_t pos = imgName.rfind('/');
            if (pos+1 != std::string::npos) {
                name = imgName.substr(pos+1);
            }

            

            std::cout<<"get add:["<<req.body<<"]"<<std::endl;
            
            cv::Mat img = cv::imread(imgName);
            
            if(img.empty())
            {
                std::cout<<"img is empty !"<<std::endl;
            }


            // 人脸检测
            std::vector<Yolov5Face_BoxStruct> face_result;
            vector<float> input;
            int ret1 =facetool.infer(img,face_result,input);   
            if(ret1==0)
            {
                cout<<"img file path:"<<imgName<<endl;
                cout<<"get face count fail:"<<face_result.size()<<endl;
                return crow::response{getGoodResp(500,"More than one facial target or has not target!")};
            }
            if(ret1==2)
            {
                cout<<"get face quality fail:"<<endl;
                return crow::response{getGoodResp(500,"img is not clear !")};
            }           

            {
                // AUTO_LOCK wlock(dbLock,0);
                ret = facedb->add(1,input);
                // fileIdMap[idIndex] = name;
                // idFileMap[name] = idIndex;
                spIdVec.push_back(idIndex);
                spIdFileNameMap[idIndex] = name;
                fileNameSpIdMap[name] = idIndex;
                
                saveFlagQue->Push(1);//saveData
                
                cout<<"add face id:["<<idIndex<<"],name:["<<name<<"]\n";
                idIndex++;
            }
            
            if(ret)
            {
                cout<<"[ERROR] db add imgs fail!"<<endl;
                return crow::response{getGoodResp(500,"db add imgs fail!")};
            }
            

        }catch(...){
            cout<<"catch error!"<<endl;
            return crow::response{getGoodResp(500,"catch error")};
        }
        
        return crow::response{getGoodIdResp(200,"success",idIndex-1)};

    });

    CROW_ROUTE(app, "/cv/face_recognition/delete/")
    .methods("POST"_method)
    ([&](const crow::request& req){
        // if(mStatus)
        //     return crow::response{getGoodResp(500,"db not init")};
        try{
            AUTO_LOCK wlock(dbLock,0);
            int ret;
            Json::Reader reader;
            Json::Value value;
            reader.parse(req.body,value);
            std::cout<<"get delete:["<<req.body<<"]"<<std::endl;
            string imgName = value["imgName"].asString();
            string name;
            size_t pos = imgName.rfind('/');
            if (pos+1 != std::string::npos) {
                name = imgName.substr(pos+1);
            }
            
           idx_t delNum = fileNameSpIdMap[name];

            if(delNum == 0)
            {
                auto it = fileNameSpIdMap.find(name);
                if(it == fileNameSpIdMap.end())
                {
                    cout<<"del id is empty,file name is ["<<name<<"]"<<endl;   
                    return crow::response{getGoodResp(200,"success")};
                }
            }
            {
                for(int i = 0; i< spIdVec.size(); i++)
                {
                    if(spIdVec[i] == delNum)
                    {
                        spIdVec.erase(spIdVec.begin() + i);
                        spIdFileNameMap.erase(delNum);
                        fileNameSpIdMap.erase(name);
                        idx_t arr[1];
                        arr[0] = i;
                        facedb->del(1,arr);
                        cout<<"index del ["<<name<<"]"<<endl;   
                        break;
                    }
                }
            }
            cout<<"del id is["<<delNum<<"],file name is ["<<name<<"]"<<endl;   
            saveFlagQue->Push(1);//saveData
        }
        catch(...)
        {
            cout<<"catch error!"<<endl;
            return crow::response{getGoodResp(500,"catch error")};
        }
        
        return crow::response{getGoodResp(200,"success")};
    });

    CROW_ROUTE(app, "/cv/face_recognition/search/")
    .methods("POST"_method)
    ([&](const crow::request& req){
        // if(mStatus)
        //     return crow::response{getGoodResp(500,"db not init")};
        AUTO_LOCK wlock(dbLock,0);
        int ret;
        Json::Reader reader;
        Json::Value value;
        reader.parse(req.body,value);
        string imgName = value["imgName"].asString();

        cv::Mat img = cv::imread(imgName);
        if(img.empty())
        {
            std::cout<<"img is empty !"<<std::endl;
        }
        std::vector<Yolov5Face_BoxStruct> face_result;
        vector<float> input;
        int ret1 =facetool.infer(img,face_result,input);   
        if(ret1==0)
        {
            cout<<"img file path:"<<imgName<<endl;
            cout<<"get face count fail:"<<face_result.size()<<endl;
            return crow::response{getGoodResp(500,"More than one facial target or has not target!")};
        }
        if(ret1==2)
        {
            cout<<"get face quality fail:"<<endl;
            return crow::response{getGoodResp(500,"img is not clear !")};
        }   

        vector<idx_t> I;
        vector<float> D;
        {
            // AUTO_LOCK rlock(dbLock,1);
            ret = facedb->search(1,1,input,I,D);
        }
        std::cout<<"get search:["<<req.body<<"]"<<std::endl;
        return crow::response{getSearchResp(I,D)};
    });

    CROW_ROUTE(app, "/cv/face_recognition/adds/")
    .methods("POST"_method)
    ([&](const crow::request& req){
        int ret;
        Json::Reader reader;
        Json::Value value;
        Json::Value retAnsArr;
        AUTO_LOCK wlock(dbLock,0);
        try{
            
            cout<<"req.body:"<<req.body<<endl;
            reader.parse(req.body,value);

            vector<string> imgNames;
            vector<cv::Mat> validFaces;
            vector<string> NameArr;
            vector<vector<Yolov5Face_BoxStruct>> face_results;
            vector<int> indexs;
            indexs.resize(0);
            
            vector<float> input;

            for(int i = 0; i < value["imgNames"].size(); i++)
            {
                Json::Value item;
                string imgName1 = value["imgNames"][i].asString();
                string name;
                size_t pos = imgName1.rfind('/');
                if (pos+1 != std::string::npos) {
                    name = imgName1.substr(pos+1);
                }
                imgNames.push_back(name);

                cv::Mat img = cv::imread(imgName1);

                // 人脸检测
                 std::vector<Yolov5Face_BoxStruct> face_result;
                // vector<float> input;
                int ret1 =facetool.infer(img,face_result,input);   
                if(ret1==0)
                {
                 cout<<"img file path:"<<imgName1<<endl;
                    cout<<"get face count fail:"<<face_result.size()<<endl;
                    // return crow::response{getGoodResp(500,"More than one facial target or has not target!")};
                    item["code"] = 500;
                    item["msg"] = "More than one facial target or has not target!";
                    retAnsArr.append(item);
                    continue;
                    }
                if(ret1==2)
                {
                 cout<<"get face quality fail:"<<endl;
                    // return crow::response{getGoodResp(500,"img is not clear !")};
                    item["code"] = 500;
                    item["msg"] = "img is not clear !";
                    retAnsArr.append(item);
                    continue;
                }          


               
                  
                
                item["code"] = 200;
                item["msg"] = "success";
                retAnsArr.append(item);
                validFaces.push_back(img);
                NameArr.push_back(name);
                face_results.push_back(face_result);
                indexs.push_back(i);

                spIdVec.push_back(idIndex);
                spIdFileNameMap[idIndex] = name;
                fileNameSpIdMap[name] = idIndex;
                retAnsArr[i]["spId"] = (int)idIndex;
                cout<<"add face id:["<<idIndex<<"],name:["<<name<<"]\n";
                idIndex++;
                
            }
        
            
            if(validFaces.size() > 0)
            {
                ret = facedb->add(validFaces.size(),input);
                saveFlagQue->Push(1);
                if(ret)
                {
                    cout<<"[ERROR] db add imgs fail!"<<endl;
                    return crow::response{getGoodResp(500,"db add imgs fail!")};
                }
            }



        }
        catch(...)
        {
            cout<<"catch error!"<<endl;
            return crow::response{getGoodResp(500,"catch error")};
        }
        Json::Value ans;
        ans["code"] = 200;
        ans["msg"] = "success";
        if(!retAnsArr.isNull())
        {ans["result"] = retAnsArr;}
        
        // saveFlagQue->Push(1);//saveData

        return crow::response{ans.toStyledString()};

    });

    CROW_ROUTE(app, "/cv/face_recognition/set/")
    .methods("POST"_method)
    ([&](const crow::request& req){
        try{
            AUTO_LOCK wlock(dbLock,0);
            int ret;
            Json::Reader reader;
            Json::Value value;
            reader.parse(req.body,value);
            std::cout<<"get set:["<<req.body<<"]"<<std::endl;
            string imgName = value["imgName"].asString();
            string name;
            size_t pos = imgName.rfind('/');
            if (pos+1 != std::string::npos) {
                name = imgName.substr(pos+1);
            }
            idx_t setNum = value["spId"].asInt();
            
            auto it = fileNameSpIdMap.find(spIdFileNameMap[setNum]);
            if(it != fileNameSpIdMap.end())
                fileNameSpIdMap.erase(it);
            fileNameSpIdMap[name] = setNum;
            spIdFileNameMap[setNum] = name;
            cout<<"set id is["<<setNum<<"],new file name is ["<<name<<"]"<<endl;   
            saveFlagQue->Push(1);//saveData
        }
        catch(...)
        {
            cout<<"catch error!"<<endl;
            return crow::response{getGoodResp(500,"catch error")};
        }
        _reload = true;
        return crow::response{getGoodResp(200,"success")};
    });

    app.port(59394).concurrency(5).run();
    
}





#ifdef STRANGERDB
void StrangeServer::save_mind_data_thread()
{
    while(1)
    {
        int flag = 0;
        if(saveFlagQue->Pop(flag) == 0)
        {
            // while(saveFlagQue->Pop(flag,100) == 0)
            // {
            //     usleep(10*1000);
            // }
            std::vector<idx_t> tmpVec = spIdVec;
            idx_t idIndexTmp = idIndex;
            Json::Value saveData;
            Json::Value jsonarr;
            saveData["newIndex"] = (int)idIndexTmp;
            
            for(auto it = tmpVec.begin(); it!= tmpVec.end();it++)
            {
                Json::Value tmp;
                tmp["id"] = (int)*it;
                jsonarr.append(tmp);
            }
            saveData["data"] = jsonarr;
            std::string saveStr = saveData.toStyledString();
            FILE * savePtr  = fopen("face_detection/FACE2.bin","wb+");
            fwrite(saveStr.c_str(),1,saveStr.size(),savePtr);
            fclose(savePtr);
            std::cout<<"[INFO] stranger save json in host!" <<std::endl;
        }
        usleep(300*1000);
    }
}

StrangeServer::StrangeServer()
{
    facetool.init("");

    FILE *fPtr = fopen("face_detection/FACE2.bin","rb");
    if(fPtr!=NULL)
    {
        string fileJson;
        char a;
        while(fread(&a,sizeof(char),1,fPtr) == 1)
        {
            fileJson.push_back(a);
        }
        Json::Reader reader;
        Json::Value value;
        reader.parse(fileJson,value);
        if(!value.isObject())
        {
            cout<<"[ERROR] Read FACE2.bin fail!\n";
        }
        else
        {
            idIndex = value["newIndex"].asInt();
            for(int i = 0; i < value["data"].size(); i++)
            {
                spIdVec.push_back(value["data"][i]["id"].asInt());
            }
        }
    }
    else
    {
        idIndex = 0;
        system("rm face_detection/facedb.bin");
    }

    mImgsPath = "face_detection/facedb.bin";
    facedb = new FaceDb(512);
    int ret = facedb->init2(mImgsPath);
    if(ret)
    {
        cout<<"[ERROR] db init imgs fail!"<<endl;
        // return ;
    }

    mStatus = true;
    saveFlagQue = new BlockingQueue<int>(9999);
    
    th_ptr2 = new std::thread(&StrangeServer::save_mind_data_thread,this);
    th_ptr = new std::thread(&StrangeServer::server_th,this);

}

StrangeServer::~StrangeServer()
{
    if(facedb)
        delete facedb;
    
}

string StrangeServer::getSearchResp(float confff,vector<idx_t> I,vector<float> D)
{
    Json::Value ans;
    Json::Value confs;
    Json::Value ids;
    
    std::cout<<"search stranger database cout:"<<D.size()<<endl;
    for(int i = 0; i < D.size();i++)
    {
        if(D[i]>confff)
            continue;
        std::cout<<"get id:"<<spIdVec[I[i]]<<","<<D[i]<<endl;
        confs.append(D[i]);
        ids.append((int)spIdVec[I[i]]);
    }

    
    ans["code"] = 200;
    ans["msg"] = "success";
    ans["confs"] = confs;
    ans["ids"] = ids;

    return ans.toStyledString();
}

void StrangeServer::server_th()
{
    crow::SimpleApp app;

    CROW_ROUTE(app, "/cv/starngers/search/")
    .methods("POST"_method)
    ([&](const crow::request& req){
        // if(mStatus)
        //     return crow::response{getGoodResp(500,"db not init")};
        AUTO_LOCK wlock(dbLock,1);
        int ret;
        Json::Reader reader;
        Json::Value value;
        reader.parse(req.body,value);
        string imgName = value["imgName"].asString();
        float conf = value["conf"].asFloat();
        int querySize = value["querySize"].asFloat();

        cv::Mat img = cv::imread(imgName);

        // 人脸检测
        std::vector<Yolov5Face_BoxStruct> face_result;
        vector<float> input;
        int ret1 =facetool.infer(img,face_result,input);   
        if(ret1==0)
        {
            return crow::response{getGoodResp(500,"More than one facial target or has not target!")};
        }
         if(ret1==2)
        {
            return crow::response{getGoodResp(500,"img is not clear !")};
        }


      
        vector<idx_t> I;
        vector<float> D;
        {
            // AUTO_LOCK rlock(dbLock,1);
            ret = facedb->search2(querySize,1,input,I,D);
        }
        std::cout<<"get search:["<<req.body<<"]"<<std::endl;
        return crow::response{getSearchResp(conf,I,D)};
    });

    CROW_ROUTE(app, "/cv/starngers/dels/")
    .methods("POST"_method)
    ([&](const crow::request& req){
        AUTO_LOCK wlock(dbLock,0);
        int ret;
        Json::Reader reader;
        Json::Value value;
        reader.parse(req.body,value);
        std::cout<<req.body<<std::endl;
        idx_t *delArr = NULL;
        if(value["ids"].size())
            delArr = (idx_t *)malloc(value["ids"].size() * sizeof(idx_t));
        for(int i = 0; i< value["ids"].size(); i++)
        {
            for(int j = 0; j< spIdVec.size(); j++) //删除中间层
            {
                if(spIdVec[j] == value["ids"][i].asInt())
                {
                    spIdVec.erase(spIdVec.begin() + j);
                    break;
                }
            }
            delArr[i] = value["ids"][i].asInt();
        }
        if(value["ids"].size())
        {
            facedb->del(value["ids"].size(), delArr); //删除数据库
            
            free(delArr);
            faiss::write_index(facedb->index,"face_detection/facedb.bin"); //数据库本地化
            saveFlagQue->Push(1);//saveData  //中间层本地化
        }
        std::cout<<"[INFO] http dels stranger ids size:"<<value["ids"].size()<<std::endl;
        return crow::response{getGoodResp(200,"success")};
    });

    app.port(49394).concurrency(5).run();
}


int StrangeServer::add(std::vector<float>& target)
{
    AUTO_LOCK wlock(dbLock,0);
    facedb->add(1,target);
    spIdVec.push_back(idIndex);
    cout<<"[INFO] add stranger face id:["<<idIndex<<"\n";
    faiss::write_index(facedb->index,"face_detection/facedb.bin");
    idIndex++;
    saveFlagQue->Push(1);//saveData
    return idIndex-1;
}

#endif