#!/usr/bin/env th
require('torch')
require('nn')
require('image')
require('paths')
require 'cunn'
require 'loadcaffe'
local ad = require('adversarial');
local ad_label = require('adversarial_label');
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(300)

LR = 1e4
epoch = 100
local eye = 224              -- small net requires 231x231
local label_nb = 286         -- label of 'bee'
local mean = 118.380948/255  -- global mean used to train overfeat
local std = 61.896913/255    -- global std used to train overfeat
local intensity = 1          -- pixel intensity for gradient sign
local choice = 'entropy'       -- 0 for minimize wrt to class, 1 for label
local copies = 10            -- number of randomly generated Gaussian pictures

local dir_path = choice..'_'..LR
dir = './' .. choice ..'_'..LR..'/'
local img_path = './images/'
local path_img = img_path..'dog.jpg'
local path_img2 = img_path..'cat.jpg'
local path_model = 'model.t7'
local file_results = io.open(dir.."results.txt", "w")
--create the directories needed
os.execute("mkdir -p " .. dir_path .. '/image+gaus')
os.execute("mkdir -p " .. dir_path .. '/image+gaus+noise')

--local label = require('overfeat_label')
local file = io.open("vgg_labels.txt", "r");
local label = {}
for line in file:lines() do
    table.insert (label, line);
end
function img_resize(path)
    -- resize input/label
    local img = image.scale(image.load(path), '^'..eye)
    local tx = math.floor((img:size(3)-eye)/2) + 1
    local ly = math.floor((img:size(2)-eye)/2) + 1
    img = img[{{},{ly,ly+eye-1},{tx,tx+eye-1}}]
    --img:add(-mean):div(std)
    --switch to BGR
    local clone = img:clone()
    img[1] = clone[3]:add(-103.939/255)
    img[2] = clone[2]:add(-116.779/255)
    img[3] = clone[1]:add(-123.68/255)
    --vgg pixel range [0,255]
    img:mul(255)
    img = img:cuda()
    return img
end

--this function converts BGR[0,255] back to RGB[0,1] for torch
function img_revert(img)
    local clone = img:clone()
    clone[1] = img[3] + 123.68
    clone[2] = img[2] + 116.779
    clone[3] = img[1] + 103.939
    clone:div(255)
    return clone
end
local imgA = img_resize(path_img)
local imgB = img_resize(path_img2)
-- get trained model (switch softmax to logsoftmax)
--local model = torch.load(path_model)
local model = loadcaffe.load('VGG_ILSVRC_19_layers_deploy.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'nn'):cuda()  
model:evaluate()
-- check prediction results
local pred = model:forward(imgA)
local val, idx = pred:max(pred:dim())

--change last layer to logsoftmax
model.modules[#model.modules] = nn.LogSoftMax()
model = model:cuda()

-- set loss function
if choice == 'MSE' then
    local loss = nn.MSECriterion():cuda() 
    noise = ad.adversarial_fast(model, loss, imgA:clone(), idx, std, intensity, copies)
else
    local loss = nn.ClassNLLCriterion():cuda()
    noise = ad_label.adversarial_fast(model, loss, imgA:clone(), label_nb, std, intensity, copies)
end
-- generate adversarial examples

--change last layer back to original softmax
model:evaluate()
model.modules[#model.modules] = nn.SoftMax()
model = model:cuda()

for i = 1, copies do
    --have to resize imgA to 4 dims for noise
    --first print add gaussian without trained noise
    local imgA_t = imgA:clone() + ad.noise(imgA:view(1, imgA:size(1), imgA:size(2), imgA:size(3)))
    --save gaussian +  image
    image.save(dir.."/image+gaus/Ad_"..i..".jpg", img_revert(imgA_t)) 
    --add the trained noise
    imgA_t:add(noise):cuda()
    local pred = model:forward(imgA_t)
    local val, idx = pred:max(pred:dim())
    print('==> adversarial:', label[ idx[1] ], 'confidence:', val[1])
    image.save(dir.."/image+gaus+noise/Ad_"..i..".jpg", img_revert(imgA_t))
    file_results:write("Image"..i..": "..label[idx[1]].." with confidence: ".. val[1], '\n')
end

local imgB_t = imgB:clone() + noise
imgB_t = imgB_t:cuda()

image.save(dir.."diff.jpg", img_revert(torch.reshape(noise, 3, eye, eye)))

print('==> original:', label[ idx[1] ], 'confidence:', val[1])
file_results:close()
--add this noise to other images to see if it affects classification
-- check prediction results
--[[local pred = model:forward(imgB)
local val, idx = pred:max(pred:dim())
print('==> original:', label[ idx[1] ], 'confidence:', val[1])

local pred = model:forward(imgB_t)
local val, idx = pred:max(pred:dim())
print('==> adversarial:', label[ idx[1] ], 'confidence:', val[1])

print('==> mean absolute diff between the original and adversarial images[min/max]:', torch.add(imgB, -imgB_t):abs():mean())
image.save("test2.jpg", imgB)
image.save("test2_t.jpg", imgB_t)]]
