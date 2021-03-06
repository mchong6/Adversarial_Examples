require('nn')
require('gnuplot')

local adversarial = {};

local function noise(input)
    local number = input:size(1)
    local dim = input:size(2)
    local w = input:size(3)
    local h = input:size(4)
    local noise = torch.Tensor(number,dim, w, h):normal(0,10)
    return noise:cuda()
end

local function adversarial_fast(model, loss, x, y, std, intensity, copies)
    parameters, gradParameters = model:getParameters()
   -- consider x as batch
	local plot = nil
    local batch = false
    if x:dim() == 3 then
        x = x:view(1, x:size(1), x:size(2), x:size(3))
        batch = true
    end
    local theta = x:clone():zero()
   --create random copies of x
    x = torch.repeatTensor(x, copies, 1, 1, 1):cuda()
    x:add(noise(x))
    
    for i = 0, epoch do
        gradParameters:zero()
        --clone x so that we dont edit original batch
        local x_batch = x:clone()
        --add theta to all x batch
        for i = 1, x:size(1) do
            x_batch[i] = x_batch[i] + theta
        end
        -- compute output
        --use Linear to select only the correct class
        model:add(nn.Linear(1000,1):cuda())
        --make sure weight is zero except for the correct class
        model.modules[#model.modules].weight:zero()   
        model.modules[#model.modules].weight[1][y[1]] = 1 
        model.modules[#model.modules].bias:zero()
        local y_hat = model:updateOutput(x_batch)
    	y_hat = {theta, y_hat}
        local temp = theta:clone():fill(0)
        y = {temp, y_hat:clone():fill(-10):cuda()}
        --show that extraction of class is correct
        --print(model.modules[#model.modules-1].output[1][y[1]])
        local f = loss:forward(y_hat, y)
        print(f)
        if plot == nil then
            plot = torch.Tensor(1):fill(f)
        else
            plot = plot:cat(torch.Tensor(1):fill(f))
        end
        local cost = loss:backward(y_hat, y_hat:clone():fill(-10):cuda())
        local x_grad = model:updateGradInput(theta, cost)
        local grad = x_grad * LR
        --remove the last layer selection
        model:remove(#model.modules)
        theta = theta - grad
    end
    gnuplot.epsfigure(dir.. 'testMSE.eps')
    gnuplot.plot('MSE',plot, '-')
    gnuplot.plotflush()
   --[[if batch then
      x = x:view(x:size(2), x:size(3), x:size(4))
   end]]
   -- return adversarial examples (inplace)
    return theta
end
adversarial.noise = noise
adversarial.adversarial_fast = adversarial_fast
return adversarial
