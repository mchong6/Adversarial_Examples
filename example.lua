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

LR = 1e6
epoch = 300 
theta_weight = 1e3
local eye = 224              -- small net requires 231x231
local label_nb = 286         -- label of 'bee'
local mean = 118.380948/255  -- global mean used to train overfeat
local std = 61.896913/255    -- global std used to train overfeat
local intensity = 1          -- pixel intensity for gradient sign
local choice = 'entropy'       -- 0 for minimize wrt to class, 1 for label
local copies = 10            -- number of randomly generated Gaussian pictures
local multi_pic = 0

local dir_path = choice..'_'..LR
dir = './' .. choice ..'_'..LR..'/'
local img_path = './images/'
local path_img = img_path..'dog.jpg'

local path_model = 'model.t7'
--create the directories needed
os.execute("mkdir -p " .. dir_path .. '/image+gaus')
os.execute("mkdir -p " .. dir_path .. '/image+gaus+noise')
local file_results = io.open(dir.."results.txt", "w")

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

local images = nil
if multi_pic == 1 then
    -- create a table of image paths
    local images_path = {}
    f = io.popen('ls '.. img_path)
    for image in f:lines() do
        table.insert(images_path, img_path..image)
    end
    --resize all images and store in a tensor.
    for key, image in pairs(images_path) do
        --resize to batch
        local temp_img = img_resize(image)
        temp_img = temp_img:view(1, temp_img:size(1), temp_img:size(2), temp_img:size(3))
        if images == nil then
            images = torch.CudaTensor(temp_img)
        else
            images = images:cat(torch.CudaTensor(temp_img), 1)
        end
    end
    images = images:cuda()
else
    images= img_resize(path_img)
    --resize image to 4dim
    images = images:view(1, images:size(1), images:size(2), images:size(3))
end

-- get trained model (switch softmax to logsoftmax)
--local model = torch.load(path_model)
local model = loadcaffe.load('VGG_ILSVRC_19_layers_deploy.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'nn'):cuda()  
model:evaluate()
-- check prediction results
local pred = model:forward(images)
local val, idx = pred:max(pred:dim())

--change last layer to logsoftmax
model.modules[#model.modules] = nn.LogSoftMax()
model = model:cuda()

-- set loss function
if choice == 'MSE' then
    noise = ad.adversarial_fast(model, loss, images:clone(), idx, std, intensity, copies)
else
    local loss = nn.ClassNLLCriterion():cuda()
    noise = ad_label.adversarial_fast(model, loss, images:clone(), label_nb, std, intensity, copies)
end
-- generate adversarial examples

--change last layer back to original softmax
model:evaluate()
model.modules[#model.modules] = nn.SoftMax()
model = model:cuda()

for j = 1, copies do
    --have to resize imgA to 4 dims for noise
    --first print add gaussian without trained noise
    local images_t = images:clone() + ad.noise(images)
    --save gaussian + image
    for i = 1, images_t:size(1) do
        image.save(dir.."/image+gaus/Pic_"..i.."_Gaus"..j..".jpg", img_revert(images_t[i])) 
    end
    --resize, repeat and add the trained noise
    --noise = noise:view(1, noise:size(1), noise:size(2), noise:size(3)) 
    noise = torch.repeatTensor(noise, images_t:size(1), 1, 1, 1)
    images_t:add(noise):cuda()
    local predict = model:forward(images_t)
    local value, index = predict:max(predict:dim())
    for i = 1, predict:size(1) do
        print('==> adversarial:', label[index[i][1] ], 'confidence:', value[i][1])
        image.save(dir.."/image+gaus+noise/Pic_"..i.."_Gaus"..j..".jpg", img_revert(images_t[i]))
        file_results:write("Image"..i..": "..label[index[i][1]].." with confidence: ".. value[i][1], '\n')
end
end

image.save(dir.."diff.jpg", img_revert(torch.reshape(noise[1], 3, eye, eye)))

for i = 1, pred:size(1) do
    print('==> original:', label[ idx[i][1] ], 'confidence:', val[i][1])
end
file_results:close()
