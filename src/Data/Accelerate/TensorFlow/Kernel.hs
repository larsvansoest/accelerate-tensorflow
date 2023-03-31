{-# LANGUAGE GADTs             #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE TypeOperators     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}


module Data.Accelerate.TensorFlow.Kernel where

import Data.Array.Accelerate.Lifetime
import Foreign
import Data.Array.Accelerate.AST.Kernel (NoKernelMetadata)
import Data.Text.Prettyprint.Doc
import Data.Array.Accelerate.Pretty.Exp
    ( prettyConst, primOperator, prettyLhs )
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Smart (typeR, undef)
import GHC.Conc (TVar(TVar))
import Data.Accelerate.TensorFlow.Operation
import Data.Array.Accelerate
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.Schedule.Uniform (BaseVar, BaseVars, fromGrounds, BaseR (BaseRground), Var (Var), mapArgs, GLeftHandSide)
import Data.Array.Accelerate.Type (ScalarType (SingleScalarType), NumType)
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.AST.Environment hiding (Val)
import Data.Array.Accelerate.AST.Partitioned
import Data.Array.Accelerate.Eval
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type (TupR(TupRsingle, TupRunit, TupRpair), Distributes (reprIsSingle), TypeR, mapTupR)
import Data.Array.Accelerate.Analysis.Match
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.Trafo.Operation.Substitution
import Data.Array.Accelerate.Pretty.Print
import Data.Accelerate.TensorFlow.Type
import qualified TensorFlow.Types as TF

data TensorKernel env where
  TensorConstant :: TF.OneOf TFAll a => a -> BaseVar env (Buffer a) -> TensorKernel env
  TensorId :: TF.OneOf TFAll a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAdd :: TF.OneOf TFNum a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorMul :: TF.OneOf TFNum a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorSub :: TF.OneOf TFNum a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorNeg :: TF.OneOf TFNeg a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAbs :: TF.OneOf TFAbs a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorSign :: TF.OneOf TFSign a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env

  TensorTruncateDiv :: TF.OneOf TFNum a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorTruncateMod :: TF.OneOf TFTruncateMod a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env

instance NFData' TensorKernel where
  rnf' !_  = ()

newtype TensorFlowKernelMetadata f =
  TensorFlowKernelMetadata { kernelArgsSize :: Int }

instance IsKernel TensorKernel where
  type KernelOperation TensorKernel = TensorOp
  type KernelMetadata  TensorKernel = NoKernelMetadata

  compileKernel :: Env AccessGroundR env -> Cluster TensorOp args -> Args env args -> TensorKernel env
  compileKernel _ cluster clusterArgs
    | ClusterOperations _ lhs [ApplyOperation operation args] <- clusterOperations cluster clusterArgs
    , Just Refl <- wildcards lhs
    = compileOperation operation args
  compileKernel _ _ _ = error "impossible, did you use SequentialSchedule?"

wildcards :: LeftHandSide a e env env' -> Maybe (env :~: env')
wildcards (LeftHandSideWildcard _)    = Just Refl
wildcards (LeftHandSidePair lhs lhs')
 | Just Refl <- wildcards lhs
 , Just Refl <- wildcards lhs'        = Just Refl
wildcards _                           = Nothing

instance PrettyKernel TensorKernel where
  prettyKernel = PrettyKernelBody True $ \_ _ -> ""

compileOperation :: TensorOp args -> Args env args -> TensorKernel env
compileOperation (TConstant stIn s) (argIn :>: _)             = compileNullaryOperation1 stIn argIn (TensorConstant s)
compileOperation (TId stIn) (argIn :>: argOut :>: _)          = compileUnaryOperation1 stIn stIn argIn argOut TensorId
        
compileOperation (TAdd stIn) (argIn :>: argOut :>: _)         = compileBinaryOperation1 stIn stIn argIn argOut TensorAdd
compileOperation (TMul stIn) (argIn :>: argOut :>: _)         = compileBinaryOperation1 stIn stIn argIn argOut TensorMul
compileOperation (TSub stIn) (argIn :>: argOut :>: _)         = compileBinaryOperation1 stIn stIn argIn argOut TensorSub
compileOperation (TNeg stIn) (argIn :>: argOut :>: _)         = compileUnaryOperation1 stIn stIn argIn argOut TensorNeg
compileOperation (TAbs stIn) (argIn :>: argOut :>: _)         = compileUnaryOperation1 stIn stIn argIn argOut TensorAbs
compileOperation (TSign stIn) (argIn :>: argOut :>: _)        = compileUnaryOperation1 stIn stIn argIn argOut TensorSign

compileOperation (TTruncateDiv stIn) (argIn :>: argOut :>: _) = compileBinaryOperation1 stIn stIn argIn argOut TensorTruncateDiv
compileOperation (TTruncateMod stIn) (argIn :>: argOut :>: _) = compileBinaryOperation1 stIn stIn argIn argOut TensorTruncateMod

compileOperation _ _ = undefined

compileNullaryOperation1 :: forall a sh env. ScalarType a -> Arg env (Out sh a) -> (BaseVar env (Buffer a) -> TensorKernel env) -> TensorKernel env
compileNullaryOperation1 stOut (ArgArray _ (ArrayR _ a) _ gvbOut) kernel
 | Refl <- reprIsSingle @ScalarType @a @Buffer stOut
 = kernel (groundToBase a gvbOut)

compileBinaryOperation1 :: forall a b sh env. ScalarType a -> ScalarType b -> Arg env (In sh (a, a)) -> Arg env (Out sh b) -> (BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer b) -> TensorKernel env) -> TensorKernel env
compileBinaryOperation1 stIn stOut (ArgArray _ (ArrayR _ (TupRpair a a')) _ (TupRpair gvbIn gvbIn')) (ArgArray _ (ArrayR _ b) _ gvbOut) kernel
 | Refl <- reprIsSingle @ScalarType @a @Buffer stIn
 , Refl <- reprIsSingle @ScalarType @b @Buffer stOut
 = kernel (groundToBase a gvbIn) (groundToBase a' gvbIn') (groundToBase b gvbOut)
compileBinaryOperation1 _ _ _ _ _ = error "impossible"

compileUnaryOperation1 :: forall a b sh env. ScalarType a -> ScalarType b -> Arg env (In sh a) -> Arg env (Out sh b) -> (BaseVar env (Buffer a) -> BaseVar env (Buffer b) -> TensorKernel env) -> TensorKernel env
compileUnaryOperation1 stIn stOut (ArgArray _ (ArrayR _ a) _ gvbIn) (ArgArray _ (ArrayR _ b) _ gvbOut) kernel
  | Refl <- reprIsSingle @ScalarType @a @Buffer stIn
  , Refl <- reprIsSingle @ScalarType @b @Buffer stOut
  = kernel (groundToBase a gvbIn) (groundToBase b gvbOut)

groundToBase :: TypeR a -> GroundVars env (Buffer a) -> BaseVar env (Buffer a)
groundToBase _ (TupRsingle (Var groundR idx)) = Var (BaseRground groundR) idx

groundsToBase :: TypeR a -> GroundVars env (Buffers a) -> BaseVars env (Buffers a)
groundsToBase _ TupRunit                              = TupRunit
groundsToBase t@(TupRsingle (st :: ScalarType e)) gvb
  | Refl <- reprIsSingle @ScalarType @e @Buffer st    = TupRsingle (groundToBase t gvb)
groundsToBase (TupRpair t1 t2) (TupRpair gvb1 gvb2)   = TupRpair (groundsToBase t1 gvb1) (groundsToBase t2 gvb2)
groundsToBase _ _                                     = error "impossible"