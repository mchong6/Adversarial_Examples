require('nn')

-- "Explaining and harnessing adversarial examples"
-- Ian Goodfellow, 2015
-- so that we can use the functions here
local adversarial_label = {};

local function noise(input)
    local number = input:size(1)
    local dim = input:size(2)
    local w = input:size(3)
    local h = input:size(4)
    local noise = torch.Tensor(number,dim, w, h):normal(0,10)
    return noise:cuda()
end

local function adversarial_fast(model, loss, x, y, std, intensity)
    parameters, gradParameters = model:getParameters()
    y = torch.CudaTensor(10):fill(y)
   -- consider x as batch
    local batch = false
    if x:dim() == 3 then
        x = x:view(1, x:size(1), x:size(2), x:size(3))
        batch = true
    end
    local theta = x:clone():zero()
   --create random copies of x
    x = torch.repeatTensor(x, 10, 1, 1, 1):cuda()
    x:add(noise(x))
    
    local add_noise = noise(x)
    for i = 0, 80 do
        gradParameters:zero()
        --clone x so that we dont edit original batch
        local x_batch = x:clone()
        --add theta to all x batch
        for i = 1, x:size(1) do
            x_batch[i] = x_batch[i] + theta
        end
        -- compute output
        --local addition = add_noise:add(x):cuda()
        local y_hat = model:updateOutput(x_batch)
    
        --show that extraction of class is correct
        --print(model.modules[#model.modules-1].output[1][y[1]])
        local f = loss:forward(y_hat, y) 
        print(f)
        local cost = loss:backward(y_hat, y) 
        local x_grad = model:updateGradInput(theta, cost)
        local grad = x_grad * 1e5
        theta = theta - grad
    end

   --[[if batch then
      x = x:view(x:size(2), x:size(3), x:size(4))
   end]]
   -- return adversarial examples (inplace)
    return theta
end
adversarial_label.noise = noise
adversarial_label.adversarial_fast = adversarial_fast
return adversarial_label
