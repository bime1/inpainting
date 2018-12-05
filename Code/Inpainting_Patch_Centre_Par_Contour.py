#!/usr/bin/env python
# coding: utf-8


import scipy.ndimage.morphology
import numpy as np
from PIL import Image
from PIL.Image import *
import matplotlib.pyplot as plt
from scipy import misc
#import ImageFilter
from PIL import Image
import pdb



# On retourne le patch Wp comme une sous image de image débutant au pixel (x,y) et de taille l
def patch(x,y,l,image):
    #Wp = image[(y):(y+l),(x):(x+l)]
    lDemi = int(np.floor(l/2))
    Wp = image[(y-lDemi):(y+lDemi+1),(x-lDemi):(x+lDemi+1)]
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

# On définit la matrice de confiance
def anti_masque(image,x_masque,y_masque,mMasque,nMasque):
    (m,n)=image.shape
    A = np.zeros((m,n))
    B = -1*np.ones((m,n))
    B[y_masque:y_masque+nMasque,x_masque:x_masque+mMasque] = 0
    A[y_masque:y_masque+nMasque,x_masque:x_masque+mMasque] = 1
    anti_matrice = scipy.ndimage.morphology.distance_transform_edt(A)
    return B +anti_matrice

# On retourne la confiance du patch centré au pixel (x,y)
def confiance(x,y,l,image,x_masque,y_masque,mMasque,nMasque,anti_masque_mat):
    confidence = 0.0
    lDemi = int(np.floor(l/2))
    for i in range(-lDemi,lDemi+1):
        for j in range(-lDemi,lDemi+1):
            if( anti_masque_mat[y+i][x+j] != -1 ):
                confidence += anti_masque_mat[y+i][x+j]
    return float(confidence/(l^2))




# On crée l'image masquée à partir de image
def image_avec_trou(image,x,y,mMasque,nMasque):
    (m,n) = image.shape
    for i in range(y,y+mMasque):
        for j in range(x,x+nMasque):
            image[j,i]=0
    misc.imsave("nouvelle image avec trou.png",image)


# On renvoie le plus proche voisin de Wp qui est un patch de longueur l
# On initialise PPV à un patch quelconque de l'image
# On initialise min_distance à une distance très très grande et on compare distance à min_distance
# La condition à la ligne 69 permet de rechercher le voisin de Wp uniquement dans l'image privée du masque

def recherche_patchvoisin(Wp,WpOcc,image,x,y,mMasque,nMasque,l):
    d = 0
    lDemi = int(np.floor(l/2))
    min_distance = 100000000
    image_occ = masque(image,x,y,mMasque, nMasque)
    matrice_image = image
    m, n = matrice_image.shape[0], matrice_image.shape[1]
    PPV = patch(2*l,2*l,l,image)
    for x in range(l + 1, n-l):
        for y in range(l + 1,m - l):
            if image_occ[(y-lDemi):(y+lDemi+1),(x-lDemi):(x+lDemi+1)].sum() == 0:
                d = distance(l,Wp,WpOcc,patch(x,y,l,image),min_distance)
                if d < min_distance:
                    PPV=patch(x,y,l,image)
                    min_distance = d

    return PPV


# On remplace le patch Wp par son plus proche voisin et on met à jour le masque image_masque
def remplace_patch(Wp,WpOcc,image,image_masque,image_confiance,l,x,y,mMasque,nMasque):
    (m,n)=image.shape
    Wp = recherche_patchvoisin(Wp,WpOcc,image,x,y,mMasque,nMasque,l)
    lDemi = int(np.floor(l/2))
    for i in range(0,l):
        for j in range(0,l):
            if WpOcc[i,j]==1:
                image[y-lDemi+i,x-lDemi+j] = Wp[i][j]
                image_masque[y-lDemi+i,x-lDemi+j] = 0
                image_confiance[y-lDemi+i,x-lDemi+j] = -1
    return image,image_masque, image_confiance

# On retourne la valeur minimale de la confiance ainsi que les coordonnées (x_min,y_min)
def min_confiance(l,image,image_confiance,x_masque,y_masque,mMasque,nMasque):
    min_confiance = 1000000000
    x_min,y_min = 0,0
    for i in range(y_masque, y_masque + nMasque):
        for j in range(x_masque, x_masque + mMasque):
            conf = image_confiance[i,j]
            if (conf < min_confiance and int(conf) != -1):
                min_confiance = conf
                x_min,y_min = j,i
    return min_confiance, x_min, y_min

# Comment lancer les tests :
def test():
    #cette ligne est pour convertir l'image en noir et blanc
    #image_grise = Image.open('peppers.png').convert('LA')
    image_couleur = misc.imread('baboon mask.png')
    #image_couleur = misc.imread('peppers mask.png')
    if(image_couleur.ndim == 2):
        image_grise = image_couleur
    elif(image_couleur.ndim == 3):

        image_grise = 0.299 * image_couleur[:,:,0] + 0.587 * image_couleur[:,:,1]  + 0.114 * image_couleur[:,:,2]
    else:
        print("Erreur, l'image n'a pas le bon nombre de dimensions")
    (p,q) = image_grise.shape
    x_masque = int(2*q/5)
    y_masque = int(2*p/5)
    mMasque = int(q/5)
    nMasque = int(p/5)

    #création du masque
    image_masque = masque(image_grise,x_masque,y_masque,mMasque,nMasque)

    # création de la confiance
    image_confiance = anti_masque(image_grise,x_masque,y_masque,mMasque,nMasque)

    l = 5
    x_variant = x_masque
    y_variant = y_masque

    lDemi = int(np.floor(l/2))


    while np.any(image_confiance + np.ones((p,q))) != 0:
        _,x_variant,y_variant = min_confiance(l,image_grise,image_confiance,x_masque,y_masque,mMasque,nMasque)
        Wp= patch(x_variant , y_variant  ,l,image_grise)
        WpOcc = patch(x_variant  , y_variant  ,l,image_masque)
        image_grise,image_masque, image_confiance = remplace_patch( Wp,WpOcc, image_grise,image_masque,image_confiance, l, x_variant, y_variant,mMasque,nMasque)
        print("En cours de traitement")
        misc.imsave("RESULTAT baboon centre par contour.png",image_grise)
        #misc.imsave("RESULTAT peppers centre par contour.png",image_grise)
        misc.imsave("image_occlusion_out.png",image_masque)
        misc.imsave("image_confiance_out.png",image_confiance)
    print("Merci pour votre patience")


#Pour lancer le test, il faut remplacer IMAGE pat le chemin de votre image
test()
