#!/usr/bin/env python
# coding: utf-8
import numpy as np
from PIL import Image
from PIL.Image import *
import matplotlib.pyplot as plt
from scipy import misc
#import ImageFilter
from PIL import Image
import pdb

#traitement Par ligne avec des Patchs non centrés
# On retourne le patch Wp comme une sous image de image débutant au pixel (x,y) et de taille l
def patch(x,y,l,image):
    Wp = image[(y):(y+l),(x):(x+l)]
    return Wp


# On retourne la distance entre 2 patchs Wp et Wq
# Le calcul de la distance entre les 2 patchs doit être incrémenté uniquement si le pixel de Wp n'est pas noir d'où la condition à la ligne 28
# WpOcc correspond au patch associé à Wp dans le masque
# On définit min_distance afin d'accélerer la recherche du patch voisin
def distance(l,Wp,WpOcc,Wq,min_distance):
    distance_carre=0
    for i in range(0,l):
        for j in range(0,l):
            if(WpOcc[i,j] == 0):
                m = float(Wp[i,j])
                n = float(Wq[i,j])
                distance_carre += (m - n)*(m -n)
                if (distance_carre>min_distance):
                    return 100000000
    return distance_carre


# On définit le masque débutant au pixel (x,y) et de longueur nMasque et de largeur mMasque
def masque(image,x,y,mMasque,nMasque):
    (m,n) = image.shape
    masque = np.zeros((m,n))
    for i in range(y,y+mMasque):
        for j in range(x,x+nMasque):
            masque[i,j] = 1
    return masque


# On crée l'image masquée à partir de image
def image_avec_trou(image,x,y,mMasque,nMasque):
    (m,n) = image.shape
    for i in range(y,y+mMasque):
        for j in range(x,x+nMasque):
            image[i,j]=0
    misc.imsave("nouvelle image avec trou.png",image)


# On renvoie le plus proche voisin de Wp qui est un patch de longueur l
# On initialise PPV à un patch quelconque de l'image
# On initialise min_distance à une distance très très grande et on compare distance à min_distance
# La condition à la ligne 69 permet de rechercher le voisin de Wp uniquement dans l'image privée du masque
def recherche_patchvoisin(Wp,WpOcc,image,x,y,mMasque,nMasque,l):
    d = 0
    min_distance = 100000000
    (p,q) = image.shape
    x_masque = int(2*q/5)
    y_masque = int(2*p/5)
    mMasque = int(q/5)
    nMasque = int(p/5)
    image_occ = masque(image,x_masque,y_masque,mMasque, nMasque)
    matrice_image = image
    m, n = matrice_image.shape[0], matrice_image.shape[1]
    PPV = patch(2*l,2*l,l,image)
    for i in range(l + 1, n-l):
        for j in range(l + 1,m - l):
            if image_occ[(i):(i+l),(j):(j+l)].sum() == 0:
                d = distance(l,Wp,WpOcc,patch(i,j,l,image),min_distance)
                if d < min_distance:
                    PPV=patch(i,j,l,image)
                    min_distance = d
    print(min_distance)
    return PPV


# On remplace le patch Wp par son plus proche voisin et on met à jour le masque image_masque
def remplace_patch(Wp,WpOcc,image,image_masque,l,x,y,mMasque,nMasque):
    (m,n)=image.shape
    Wp = recherche_patchvoisin(Wp,WpOcc,image,x,y,mMasque,nMasque,l)
    for i in range(0,l):
        for j in range(0,l):
            if WpOcc[i,j]==1:
                image[y+i,x+j] = Wp[i][j]
                image_masque[y+i,x+j] = 0
    return image,image_masque

def test():
    #image_grise = Image.open('peppers.png').convert('LA')
    image_couleur = misc.imread('baboon mask.png')
    #image_couleur = misc.imread('peppers mask.png')
    if(image_couleur.ndim == 2):
        image_grise = image_couleur
    elif(image_couleur.ndim == 3):

        image_grise = 0.299 * image_couleur[:,:,0] + 0.587 * image_couleur[:,:,1]  + 0.114 * image_couleur[:,:,2]
    else:
        print("Erreur, l'image n'a pas le bon nombre de dimensions")
        return
    (p,q) = image_grise.shape
    x_masque = int(2*q/5)
    y_masque = int(2*p/5)
    mMasque = int(q/5)
    nMasque = int(p/5)

    #création du masque
    image_masque = masque(image_grise,x_masque,y_masque,mMasque,nMasque)
    l = 3
    x_variant = x_masque-l+1
    y_variant = y_masque-l+1
    while(y_variant < y_masque + nMasque):
        while ( x_variant < x_masque + mMasque):
            Wp= patch(x_variant  , y_variant  ,l,image_grise)
            WpOcc = patch(x_variant  , y_variant  ,l,image_masque)
            image_grise,image_masque = remplace_patch( Wp,WpOcc, image_grise,image_masque, l, x_variant, y_variant,mMasque,nMasque)
            x_variant = x_variant + 1
            print("En cours de traitement")
            misc.imsave("RESULTAT baboon non centre par ligne.png",image_grise)
            #misc.imsave("RESULTAT peppers non centre par ligne.png",image_grise)
            misc.imsave("image_masque.png",image_masque)
        x_variant = x_masque
        y_variant = y_variant + 1
    print("Merci pour votre patience")




#Pour lancer le test, il faut remplacer IMAGE pat le chemin de votre image
test()
