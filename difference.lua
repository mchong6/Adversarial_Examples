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

cmd = torch.CmdLine()
cmd:option('-copies', 10)
params = cmd:parse(arg)

LR = 5e5
epoch = 150
theta_weight = 1e1
local eye = 224              -- small net requires 231x231
local label_nb = 45          -- label of 'bee'
local mean = 118.380948/255  -- global mean used to train overfeat
local std = 61.896913/255    -- global std used to train overfeat
local intensity = 1          -- pixel intensity for gradient sign
local choice = 'entropy'       -- 0 for minimize wrt to class, 1 for label
--local copies = 10            -- number of randomly generated Gaussian pictures
local multi_pic = 1

local dir_path = choice..'_'..LR..'_'..params.copies..'copies'
dir = './' .. choice ..'_'..LR..'_'..params.copies..'copies/'
local img_path = './images/'
local path_img = img_path..'cat.jpg'
local path_gaus = dir..'/image+gaus/'
local path_adv = dir..'/image+gaus+noise/'

local path_model = 'model.t7'
local model_layers = io.open(dir.."layers.txt", "w")

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
local images2 = nil
if multi_pic == 1 then
    -- create a table of image paths
    local gaus_table = {}
    local adv_table = {}
    f = io.popen('ls '.. path_gaus)
    f2 = io.popen('ls '.. path_adv)

    for image in f:lines() do
        table.insert(gaus_table, path_gaus..image)
    end
    for image in f2:lines() do
        table.insert(adv_table, path_adv..image)
    end
    --resize all images and store in a tensor.
    for key, image in pairs(gaus_table) do
        --resize to batch
        local temp_img = img_resize(image)
        temp_img = temp_img:view(1, 1, temp_img:size(1), temp_img:size(2), temp_img:size(3))
        if images == nil then
            images = torch.CudaTensor(temp_img)
        else
            images = images:cat(torch.CudaTensor(temp_img), 1)
        end
    end
    --resize all images and store in a tensor.
    for key, image in pairs(adv_table) do
        --resize to batch
        local temp_img = img_resize(image)
        temp_img = temp_img:view(1, 1, temp_img:size(1), temp_img:size(2), temp_img:size(3))
        if images2 == nil then
            images2 = torch.CudaTensor(temp_img)
        else
            images2 = images2:cat(torch.CudaTensor(temp_img), 1)
        end
    end
    images = images:cat(images2, 2)
else
    local image_gaus= img_resize(path_gaus)
    local image_adv = img_resize(path_adv)
    --concatenate the two images
    images = image_gaus:cat(image_adv, 1) 
end
--resize image to 4dim
--image_gaus = image_gaus:view(1, image_gaus:size(1), image_gaus:size(2), image_gaus:size(3))
--image_adv = image_adv:view(1, image_adv:size(1), image_adv:size(2), image_adv:size(3))

-- get trained model 
local model = loadcaffe.load('VGG_ILSVRC_19_layers_deploy.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'nn'):cuda()  
model:evaluate()

local loss = nn.MSECriterion():cuda() 
local plot = torch.Tensor(#model.modules):fill(0)
-- check prediction results
--local val, idx = pred:max(pred:dim())
--get output from each layer
for i = 1, images:size(1) do
    local pred = model:forward(images[i])
    for j = 1, #model.modules do
        local output = model.modules[j].output
        --[[print(#output)
        print(#output[1])
        print(#output[2])]]
        --sum up error
        error = loss:forward(output[1], output[2])    
        --normalize by number of outputs
        error = error / output[1]:nElement() 
        plot[j] = plot[j] + error
    end
end

for i = 1, #model.modules do
    model_layers:write(i .. " " ..tostring(model.modules[i]) .. '\n')
end

plot = plot:div(images:size(1))
gnuplot.epsfigure(dir.. 'outputdiff.eps')
gnuplot.plot({'MSE',plot, '-'})
gnuplot.plotflush()
model_layers:close()
