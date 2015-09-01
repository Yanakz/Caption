# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:58:48 2015

@author: haoran
"""

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import pickle
import os
#import nltk
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import scipy.io as sio
import datetime
import math
import scipy.spatial as sp


class CaptionSaliency:
    def __init__(self,dataType,usingSet,dataDir,savefileDir):
        #setpath
        self.dataType = dataType
        self.usingSet = usingSet
        self.dataDir = dataDir
        self.savefileDir = savefileDir
        self.InsFile='%s/annotations/instances_%s.json'%(dataDir,dataType)
        self.CapFile='%s/annotations/captions_%s.json'%(dataDir,dataType)
        

        self.SALICON = pickle.load(open('%s/%s.p'%(savefileDir,usingSet),'rb'))
        self.Ins_ID = pickle.load(open('%s/Ins_ID_%s.p'%(savefileDir,usingSet),'rb'))
        
        self.category = pickle.load(open('%s/category.p'%savefileDir,'rb'))

        self.category_idx = pickle.load(open('%s/cat_dict_idx.p'%savefileDir,'rb'))#eg., person -- 1
        self.category_supercategory_idx = pickle.load(open('%s/cat_dict_supercat.p'%savefileDir,'rb')) #eg., person--human 
        self.supercategory_idx = pickle.load(open('%s/supercate_id.p'%savefileDir,'rb'))#eg., food--1
       
        
        self.Ins_coco = COCO(self.InsFile)
        self.Cap_coco = COCO(self.CapFile)
        self.cat_list = self.Ins_coco.cats#category list  (official)
        
        wordmat = sio.loadmat('%s/word_mat_%s.mat'%(savefileDir,usingSet))
        wordmat = wordmat['word_mat']
        self.wordmat = wordmat[:,0]
        
        
        self.correction_list = ['men','man','kid','boy','baby']
        
        self.nounlist = []
        self.nounID = []
        self.Cardi_Noun = []
        self.Seque_Noun = []
        
        
        self.size_norm = float(640*480)
        self.loc_norm = float(math.sqrt(640**2+480**2)) 
        
        self.saliencydict_c = {}
        self.saliencydict_s = {}
        
    def show_im(self,image_id):
        if image_id == None:
            raise NameError('no image ID')
        I = io.imread('%s/images/%s/%s'%(self.dataDir,self.dataType,self.SALICON['SALICON_filename'][image_id]))
        plt.imshow(I)
    
    def show_ann(self,image_id):
        if image_id == None:
            raise NameError('no image ID')
        blankim = np.zeros((480,640,3),np.uint8)
        plt.imshow(blankim)
        annIds = self.Ins_coco.getAnnIds(self.SALICON['SALICON_id'][image_id])
        anns = self.Ins_coco.loadAnns(annIds)
        self.Ins_coco.showAnns(anns)
        
    def show_cap(self,image_id):
        if image_id == None:
            raise NameError('no image ID')
        annIds = self.Cap_coco.getAnnIds(self.SALICON['SALICON_id'][image_id])
        anns = self.Cap_coco.loadAnns(annIds)
        self.Cap_coco.showAnns(anns)
    
    
    
    def findID(self,word,im_idd):
        if word in self.category:
            return self.category_idx[word]
        else:
            temp_idlist={}
            for item in self.category_idx.keys():
                for item1 in wn.synsets(item, wn.NOUN):
                    for word1 in wn.synsets(word, wn.NOUN):
                        dist = item1.wup_similarity(word1)
                        if item not in temp_idlist.keys():
                            temp_idlist[self.category_idx[item]] = dist
                            continue
                        if dist > temp_idlist[self.category_idx[item]]:
                            temp_idlist[self.category_idx[item]] = dist
            temp_idlist = sorted(temp_idlist.iteritems(), key=lambda d: d[1], reverse = True)
            temp_idlist = temp_idlist[0:1]
            for n in temp_idlist:
                if n[0] in self.Ins_ID[im_idd]:
                    return n[0]
            return 0
         
         
         
         
    
    def initial_val(self):
        if os.path.isfile(self.savefileDir+'/'+self.usingSet+'_Cardi.p') \
        and os.path.isfile(self.savefileDir+'/'+self.usingSet+'_Seque.p'):
            self.nounlist = pickle.load(open('%s/%s_Cardi.p'%(self.savefileDir,self.usingSet),'rb'))
            self.nounID = pickle.load(open('%s/%s_Seque.p'%(self.savefileDir,self.usingSet),'rb'))
        else:
            self.Cardi_Noun = []  
            self.Seque_Noun = []
            for group in self.nounID:
                 imdict = {}
                 cardi=[]
                 for item in group:
                     if not item:
                         continue
                     for idx in item:
                         cardi.append(idx)
                 u_set = list(set(cardi))
                 n_obj = len(u_set)
                 for uitem in u_set:
                     num = cardi.count(uitem)
                     imdict[uitem] = num
                 imdict= sorted(imdict.iteritems(), key=lambda d:d[1], reverse = True)
                 self.Cardi_Noun.append(imdict)
                 
                 seque={}
                 seq = [0]*n_obj
                 iid = 0
                 for iid, item in enumerate(u_set):
                     for imseq in group:
                         if not imseq or item not in imseq:
                             continue
                         wid = imseq.index(item)
                         if type(wid)==list:
                             for wwid in wid:
                                 seq[iid]+=n_obj/(wwid+1)
                         else:
                             seq[iid]+=n_obj/(wid+1)
                     seque[item] = seq[iid]
                 seque= sorted(seque.iteritems(), key=lambda d:d[1], reverse = True)  
                 self.Seque_Noun.append(seque)
            

#        try:
#            self.nounlist = pickle.load(open('%s/%s_nounlist.p'%(self.dataType,self.savefileDir),'rb'))
#        except pickle.PickleError as per:  
#            print("Pickel Error:"+str(per)) 
    def compute_distance(self):     
        if os.path.isfile(self.savefileDir+'/'+self.usingSet+'_nounlist.p') \
        and os.path.isfile(self.savefileDir+'/'+self.usingSet+'_nounID.p'):
            print('data already exist, loading...')
            self.nounlist = pickle.load(open('%s/%s_nounlist.p'%(self.savefileDir,self.usingSet),'rb'))
            self.nounID = pickle.load(open('%s/%s_nounID.p'%(self.savefileDir,self.usingSet),'rb'))
            self.initial_val()            
            print('caption saliency value loaded...!')
        else:
            time_t = datetime.datetime.utcnow()
            print('begin to compute distance...')
            print('this may take a couple of hours...')
            print('progress will be printed after an interval of 1000 images')
            self.nounlist = []
            self.nounID = []
            for im_id, captions_perim in enumerate(self.wordmat):
                noun_im = []
                nounID_im = []
                for caption in captions_perim:
                    noun_perst = []
                    nounid_perst = []
                    for noun in caption[0]:
                       word =  (noun.item())[0]
                       word = wn.morphy(word, wn.NOUN)
                       if word is None:
                           continue
                       I_ID = 0
                       if word in self.correction_list:
                           I_ID = 1
                       else:
                           I_ID = self.findID(word, im_id)
                       if I_ID == 0:
                           continue
                       noun_perst.append(word)
                       nounid_perst.append(I_ID) 
                    noun_im.append(noun_perst)
                    nounID_im.append(nounid_perst)     
                self.nounlist.append(noun_im)
                self.nounID.append(nounID_im)
                if im_id%1000==0:
                    print im_id
            self.initial_val()
            print datetime.datetime.utcnow() - time_t
            print('saving data...!')
            pickle.dump(self.nounlist,open('data/%s_nounlist.p'%self.usingSet,'wb'))
            pickle.dump(self.nounID,open('data/%s_nounID.p'%self.usingSet,'wb'))
            pickle.dump(self.Cardi_Noun,open('data/%s_Cardi_Noun.p'%self.usingSet,'wb'))
            pickle.dump(self.Seque_Noun,open('data/%s_Seque_Noun.p'%self.usingSet,'wb'))
            
            print('caption saliency value computed...!')
    
    def assign_value(self,init_val=None): 
        [saliency_dict_Cardi,saliency_dict_Seque] = self.init_salient()     
        self.saliencydict_c = saliency_dict_Cardi
        self.saliencydict_s = saliency_dict_Seque
#==============================================================================
        if not init_val == None:
            idlist = self.SALICON['SALICON_id']
            for im_id in range(len(idlist)):
                annIds = self.Ins_coco.getAnnIds(idlist[im_id])
                anns = self.Ins_coco.loadAnns(annIds)
                initv = init_val[im_id]
                for item in anns:
                    if self.saliencydict_c[im_id][item['id']] is not 0:                      
                        v = self.saliencydict_c[im_id][item['id']]
                        v = v+initv[item['id']]['size']-initv[item['id']]['dtc']
                        self.saliencydict_c[im_id][item['id']] = v
                    if self.saliencydict_s[im_id][item['id']] is not 0:
                        v = self.saliencydict_s[im_id][item['id']]
                        v = v+initv[item['id']]['size']-initv[item['id']]['dtc']
                        self.saliencydict_s[im_id][item['id']] = v
#==============================================================================
    
    def init_salient(self):
        #turn list to dict
        dict_Cardi = {}
        dict_Seque = {}
        for im_id, [item_c,item_s] in enumerate(zip(self.Cardi_Noun,self.Seque_Noun)):
            dictc = {}
            dicts = {}
            if not not item_c:
                for item in item_c:
                    dictc[item[0]] = item[1] 
            if not not item_s:
                for item in item_s:
                    dicts[item[0]] = item[1]  
            dict_Cardi[im_id] = dictc
            dict_Seque[im_id] = dicts
             
        #calc
        saliency_dict_Cardi = {}
        saliency_dict_Seque = {}
        idlist = self.SALICON['SALICON_id']
        for im_id in range(len(idlist)):
            annIds = self.Ins_coco.getAnnIds(idlist[im_id])
            anns = self.Ins_coco.loadAnns(annIds)
            sa_dict_Cardi = {}
            sa_dict_Seque = {}
            for item in anns:
                sa_dict_Cardi[item['id']] = 0
                sa_dict_Seque[item['id']] = 0
                if item['category_id'] in dict_Cardi[im_id].keys():
                    sa_dict_Cardi[item['id']] = dict_Cardi[im_id][item['category_id']]           
                if item['category_id'] in dict_Seque[im_id].keys():
                    sa_dict_Seque[item['id']] = dict_Seque[im_id][item['category_id']]
            saliency_dict_Cardi[im_id] = sa_dict_Cardi
            saliency_dict_Seque[im_id] = sa_dict_Seque
        return saliency_dict_Cardi,saliency_dict_Seque
                    
                
                    
                
                
                
    #entrance function for saliency calculation  
    def factored(self):
        factors = {}
        #calculate 3 factors:size, location, density    
        size,loc, den, dtc= [],[],[],[]
        print('Start to calculate factors...')
        for im_id in range(len(self.SALICON['SALICON_filename'])):            
            annIds = self.Ins_coco.getAnnIds(self.SALICON['SALICON_id'][im_id])
            anns = self.Ins_coco.loadAnns(annIds)           
            size.append(self.factored_size(anns))
            loc.append(self.factored_loc(anns))
            den.append(self.factored_den(anns))
            dtc.append(self.factored_dtc(anns))
        
        count = 0
        for sz,lc,de,dc in zip(size,loc,den,dtc):
            fac = {}
            for anid in sz.keys():
                fac[anid] = {}
                fac[anid]['size'] = sz[anid]
                fac[anid]['location'] = lc[anid]
                fac[anid]['density'] = de[anid]
                fac[anid]['dtc'] = dc[anid]
            factors[count] = fac
            count+=1
        print('Done!.')
        print('Assigning salient value...')
        self.assign_value(factors)
        print('salient value computed.!')
        
        
        
            
    
    def factored_size(self,ann):
        im_sz = {}        
        for item in ann:
            sz = round(item['area']/self.size_norm,2)
            im_sz[item['id']] = sz
        return im_sz 
    
    def factored_loc(self,ann):
        im_loc = {}
        for item in ann:
            loc = item['bbox']
            loc = round(loc[0]+loc[2]/2)+640*(round(loc[1]+loc[3]/2)-1)
            loc = loc/self.size_norm
            im_loc[item['id']] = loc
        return im_loc
    
    def factored_den(self,ann):
        im_den = {}
        distmat = []
        for item in ann:
            coord = item['bbox']
            c_coord = [coord[0]+coord[2]/2,coord[1]+coord[3]/2]
            distmat.append(c_coord)
        for item1,item2 in zip(distmat,ann):
            xa = [item1]
            den = sp.distance.cdist(xa,distmat)
            den = den.mean()
            im_den[item2['id']] = den
        return im_den    
    def factored_dtc(self,ann):
        im_dtc = {}
        c = [320,240]
        for item in ann:
            coord = item['bbox']
            c_coord = [coord[0]+coord[2]/2,coord[1]+coord[3]/2]
            d = sp.distance.pdist([c,c_coord])
            d = d[0]
            d = d*2/self.loc_norm
            im_dtc[item['id']] = d
        return im_dtc
        
            
           
            